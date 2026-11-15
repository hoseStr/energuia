from datetime import UTC, date, datetime

import pandas as pd
import pytest

from demand_forecast import config
from demand_forecast.data import store
from demand_forecast.data.xm_client import XMUnavailableError
from demand_forecast.models.backtest import History
from demand_forecast.models.registry import RegistryError, register_model
from demand_forecast.models.train import LightGBMForecaster
from demand_forecast.monitoring.evaluate import metrics_path, read_metrics
from demand_forecast.pipeline import StaleDataError, run_daily
from demand_forecast.serving.log import predictions_dir, read_log

BOGOTA = config.TIMEZONE
FIRST_DAY = "2025-07-01"


def day(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=BOGOTA)


class FakeXM:
    """Stand-in for `XMClient`: each series is published up to its own last day."""

    def __init__(self, make_wide, demand_until: str, generation_until: str) -> None:
        self._make_wide = make_wide
        self.until = {
            config.DEMAND.metric_id: demand_until,
            config.GENERATION.metric_id: generation_until,
        }
        self.down = False

    def publish(self, demand_until: str, generation_until: str) -> None:
        self.until = {
            config.DEMAND.metric_id: demand_until,
            config.GENERATION.metric_id: generation_until,
        }

    def fetch_hourly(self, metric_id: str, entity: str, start: date, end: date) -> pd.DataFrame:
        if self.down:
            raise XMUnavailableError("XM did not answer DemaCome/Sistema after 4 attempts")
        days = (pd.Timestamp(self.until[metric_id]) - pd.Timestamp(FIRST_DAY)).days + 1
        table = self._make_wide(FIRST_DAY, days)
        dates = table["date"].dt.date
        return table[(dates >= start) & (dates <= end)].reset_index(drop=True)


@pytest.fixture
def dirs(tmp_path):
    return tmp_path / "data", tmp_path / "models"


@pytest.fixture
def xm(make_wide) -> FakeXM:
    """XM on 2026-11-14: demand complete up to D-6 and generation up to D-4."""
    return FakeXM(make_wide, "2026-11-08", "2026-11-10")


@pytest.fixture
def registered(dirs, make_series) -> str:
    """A small model registered as the current one."""
    _, models_dir = dirs
    series = make_series(FIRST_DAY, 480)
    model = LightGBMForecaster(rounds=10, train_years=1)
    model.fit(History(demand=series, generation=series))
    entry = register_model(
        model,
        trained_at=datetime(2026, 11, 13, 16, 30, 5, tzinfo=UTC),
        data={"fingerprint": "0123456789abcdef"},
        backtest={"start": "2025-11-01", "end": "2026-10-31", "folds": 12, "metrics": {}},
        commit={"commit": "abc123", "dirty": False},
        models_dir=models_dir,
    )
    return entry["version"]


def run(xm, dirs, now: str):
    data_dir, models_dir = dirs
    return run_daily(xm, now=day(now), data_dir=data_dir, models_dir=models_dir)


def test_a_run_ingests_and_logs_the_forecast_for_tomorrow(xm, dirs, registered):
    report = run(xm, dirs, "2026-11-14 15:17")

    data_dir, _ = dirs
    assert report.issued
    assert report.forecast.target_day == day("2026-11-15")
    assert report.forecast.model_version == registered
    assert report.forecast.demand_last_day == day("2026-11-08")
    assert report.forecast.generation_last_day == day("2026-11-10")
    assert report.evaluated_days == 0
    assert report.recent_mape is None
    assert [one.series for one in report.ingest] == ["demand", "generation"]
    log = read_log(data_dir)
    assert len(log) == 24
    assert set(log["target_day"]) == {day("2026-11-15")}
    assert set(log["issued_at"]) == {day("2026-11-14 15:17")}
    assert store.load_series("demand", data_dir).index.max() == day("2026-11-08 23:00")


def test_a_second_run_on_the_same_day_does_not_touch_the_log(xm, dirs, registered):
    run(xm, dirs, "2026-11-14 15:17")
    data_dir, _ = dirs
    path = predictions_dir(data_dir) / "2026.csv"
    before = path.read_bytes()

    report = run(xm, dirs, "2026-11-14 21:40")

    assert not report.issued
    assert report.forecast.target_day == day("2026-11-15")
    assert report.forecast.issued_at == day("2026-11-14 15:17")
    assert path.read_bytes() == before


def test_forecasts_are_evaluated_once_their_demand_is_published(xm, dirs, registered):
    data_dir, _ = dirs
    run(xm, dirs, "2026-11-14 15:17")
    xm.publish("2026-11-09", "2026-11-11")
    report = run(xm, dirs, "2026-11-15 15:17")
    assert report.evaluated_days == 0
    assert read_metrics(data_dir).empty

    xm.publish("2026-11-16", "2026-11-18")
    report = run(xm, dirs, "2026-11-22 15:17")

    assert report.evaluated_days == 2
    daily = read_metrics(data_dir)
    assert list(daily["target_day"]) == [day("2026-11-15"), day("2026-11-16")]
    assert (daily["mape_percent"] > 0).all()
    assert report.recent_mape is None
    assert list(read_log(data_dir)["target_day"].unique()) == [
        day("2026-11-15"),
        day("2026-11-16"),
        day("2026-11-23"),
    ]


def test_a_day_without_a_run_stays_out_of_the_log(xm, dirs, registered):
    run(xm, dirs, "2026-11-14 15:17")
    xm.publish("2026-11-10", "2026-11-12")

    run(xm, dirs, "2026-11-16 15:17")

    data_dir, _ = dirs
    assert list(read_log(data_dir)["target_day"].unique()) == [day("2026-11-15"), day("2026-11-17")]


def test_when_xm_is_down_nothing_is_written(xm, dirs, registered):
    xm.down = True

    with pytest.raises(XMUnavailableError):
        run(xm, dirs, "2026-11-14 15:17")

    data_dir, _ = dirs
    assert not data_dir.exists()


def test_when_xm_goes_down_later_the_log_and_the_data_stay_as_they_were(xm, dirs, registered):
    run(xm, dirs, "2026-11-14 15:17")
    data_dir, _ = dirs
    before = {path: path.read_bytes() for path in data_dir.rglob("*") if path.is_file()}
    xm.down = True

    with pytest.raises(XMUnavailableError):
        run(xm, dirs, "2026-11-15 15:17")

    assert {path: path.read_bytes() for path in data_dir.rglob("*") if path.is_file()} == before


def test_no_forecast_is_issued_when_the_demand_is_too_old(xm, dirs, registered):
    limit = config.DATA_LAG_DAYS + config.MAX_EXTRA_LAG_DAYS
    late = day("2026-11-08") + pd.Timedelta(days=limit + 1, hours=15)

    with pytest.raises(StaleDataError, match="demand is published up to 2026-11-08"):
        run(xm, dirs, late.strftime("%Y-%m-%d %H:%M"))

    data_dir, _ = dirs
    assert read_log(data_dir).empty
    assert metrics_path(data_dir).exists()


def test_a_forecast_is_still_issued_at_the_limit_of_how_old_the_data_can_be(xm, dirs, registered):
    limit = config.DATA_LAG_DAYS + config.MAX_EXTRA_LAG_DAYS
    on_time = day("2026-11-08") + pd.Timedelta(days=limit, hours=15)
    generation_until = on_time.normalize() - pd.Timedelta(days=config.GENERATION_LAG_DAYS)
    xm.publish("2026-11-08", generation_until.strftime("%Y-%m-%d"))

    report = run(xm, dirs, on_time.strftime("%Y-%m-%d %H:%M"))

    assert report.issued
    assert report.forecast.demand_last_day == day("2026-11-08")


def test_no_forecast_is_issued_when_the_generation_is_too_old(xm, dirs, registered):
    xm.publish("2026-11-20", "2026-11-10")
    limit = config.GENERATION_LAG_DAYS + config.MAX_EXTRA_LAG_DAYS
    late = day("2026-11-10") + pd.Timedelta(days=limit + 1, hours=15)

    with pytest.raises(StaleDataError, match="generation is published up to 2026-11-10"):
        run(xm, dirs, late.strftime("%Y-%m-%d %H:%M"))


def test_without_a_registered_model_the_log_is_not_created(xm, dirs):
    with pytest.raises(RegistryError, match="run the `train` command first"):
        run(xm, dirs, "2026-11-14 15:17")

    data_dir, _ = dirs
    assert not predictions_dir(data_dir).exists()
