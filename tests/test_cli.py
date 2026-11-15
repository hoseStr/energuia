from datetime import UTC, date, datetime

import pandas as pd
import pytest
from typer.testing import CliRunner

from demand_forecast import cli, config
from demand_forecast.data import store
from demand_forecast.data.ingest import ingest
from demand_forecast.data.xm_client import XMUnavailableError
from demand_forecast.models.backtest import History
from demand_forecast.models.promote import Comparison
from demand_forecast.models.registry import load_model, read_registry, register_model
from demand_forecast.models.train import LightGBMForecaster
from demand_forecast.monitoring.drift import DriftSummary
from demand_forecast.monitoring.evaluate import write_metrics
from demand_forecast.serving.log import read_log

runner = CliRunner()


class FakeXM:
    """Stand-in for `XMClient` that serves one synthetic table for every metric."""

    table: pd.DataFrame | None = None

    def fetch_hourly(self, metric_id: str, entity: str, start: date, end: date) -> pd.DataFrame:
        if self.table is None:
            raise XMUnavailableError("XM did not answer DemaCome/Sistema after 4 attempts")
        days = self.table["date"].dt.date
        return self.table[(days >= start) & (days <= end)].reset_index(drop=True)


class QuickModel(LightGBMForecaster):
    """The configured model with few trees, to keep the tests fast."""

    def __init__(self) -> None:
        super().__init__(rounds=10)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(
        config, "MLFLOW_TRACKING_URI", "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
    )
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.setattr(cli, "XMClient", FakeXM)
    monkeypatch.setattr(cli, "LightGBMForecaster", QuickModel)
    monkeypatch.setattr(FakeXM, "table", None)
    return data_dir


def tracked_runs() -> pd.DataFrame:
    import mlflow

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    return mlflow.search_runs(experiment_names=[config.MLFLOW_EXPERIMENT])


@pytest.fixture
def two_month_windows(monkeypatch):
    """Evaluate on 2026-09-01..2026-10-31 and select on 2026-07-01..2026-08-31: much faster."""
    monkeypatch.setattr(config, "BACKTEST_MONTHS", 2)


@pytest.fixture
def stored_history(data_dir, make_wide, monkeypatch):
    """Synthetic data from 2025-07-01 to 2026-11-06, already ingested."""
    monkeypatch.setattr(FakeXM, "table", make_wide("2025-07-01", 494))
    ingest(
        FakeXM(),
        today=date(2026, 11, 11),
        now=datetime(2026, 11, 12, 3, 0, tzinfo=UTC),
        data_dir=data_dir,
    )


def test_backtest_prints_every_model_and_the_breakdowns(stored_history):
    result = runner.invoke(cli.app, ["backtest", "--model", "seasonal-naive"])

    assert result.exit_code == 0, result.output
    assert "from 2025-11-01 to 2026-10-31: 12 monthly folds" in result.output
    table = result.output.split("seasonal-naive by day type")[0]
    assert "seasonal-naive" in table
    assert "moving-average" in table
    assert "lightgbm" in table
    assert "MAPE (%)" in table
    assert "sunday_or_holiday" in result.output
    assert "seasonal-naive by hour of the day" in result.output
    by_month = result.output.split("seasonal-naive by month")[1]
    assert "2025-11" in by_month
    assert "2026-10" in by_month
    worst = result.output.split("seasonal-naive: the 10 days with the largest error")[1]
    assert len(worst.strip().splitlines()) == 11


def test_backtest_can_break_down_the_model(stored_history, two_month_windows):
    result = runner.invoke(cli.app, ["backtest", "--model", "lightgbm"])
    assert result.exit_code == 0, result.output
    assert "from 2026-09-01 to 2026-10-31: 2 monthly folds" in result.output
    assert "lightgbm by day type" in result.output


def test_train_registers_the_model_with_its_backtest_and_tracks_the_run(
    stored_history, two_month_windows
):
    result = runner.invoke(cli.app, ["train"])

    assert result.exit_code == 0, result.output
    assert "Registered lightgbm-" in result.output
    model, entry = load_model(config.MODELS_DIR)
    assert model.booster is not None
    assert entry["data"]["demand_last_day"] == "2026-11-06"
    assert entry["data"]["last_target_day"] == "2026-11-06"
    assert entry["backtest"]["start"] == "2026-09-01"
    assert entry["backtest"]["end"] == "2026-10-31"
    assert entry["backtest"]["folds"] == 2
    assert set(entry["backtest"]["metrics"]) == {"seasonal-naive", "moving-average", "lightgbm"}
    assert set(entry["backtest"]["metrics"]["lightgbm"]) == {"mae", "mape"}

    (run,) = tracked_runs().to_dict(orient="records")
    assert run["tags.stage"] == "train"
    assert run["tags.model_version"] == entry["version"]
    assert run["tags.data_version"] == entry["data"]["fingerprint"]
    assert run["params.rounds"] == "10"
    registered_mape = entry["backtest"]["metrics"]["lightgbm"]["mape"]
    assert run["metrics.mape"] == pytest.approx(registered_mape, abs=1e-4)
    assert "metrics.mape_moving-average" in run


def test_train_without_data_registers_nothing(data_dir):
    result = runner.invoke(cli.app, ["train"])
    assert result.exit_code == 1
    assert "run the `ingest` command first" in result.output
    assert read_registry(config.MODELS_DIR) == {"current": None, "models": []}


def test_tune_scores_the_grid_on_the_months_before_the_evaluation_window(
    stored_history, two_month_windows, monkeypatch
):
    monkeypatch.setattr(config, "TUNING_GRID", {"rounds": [5], "num_leaves": [4, 8]})

    result = runner.invoke(cli.app, ["tune"])

    assert result.exit_code == 0, result.output
    assert "from 2026-07-01 to 2026-08-31: 2 monthly folds" in result.output
    assert "{'rounds': 5, 'num_leaves': 4}: MAPE" in result.output
    assert "{'rounds': 5, 'num_leaves': 8}: MAPE" in result.output
    runs = tracked_runs()
    assert len(runs) == 2
    assert set(runs["tags.stage"]) == {"tune"}
    assert set(runs["tags.window"]) == {"2026-07-01..2026-08-31"}
    assert set(runs["params.num_leaves"]) == {"4", "8"}


def test_tune_with_too_little_history_fails_cleanly(stored_history):
    result = runner.invoke(cli.app, ["tune"])
    assert result.exit_code == 1
    assert "error:" in result.output


def test_backtest_details_follow_the_chosen_model(stored_history, two_month_windows):
    result = runner.invoke(cli.app, ["backtest", "--model", "moving-average"])
    assert result.exit_code == 0, result.output
    assert "moving-average by day type" in result.output


def test_backtest_without_data_says_how_to_get_it(data_dir):
    result = runner.invoke(cli.app, ["backtest"])
    assert result.exit_code == 1
    assert "run the `ingest` command first" in result.output


def test_unknown_model_is_rejected(stored_history):
    result = runner.invoke(cli.app, ["backtest", "--model", "prophet"])
    assert result.exit_code == 2


def test_ingest_reports_what_changed(data_dir, make_wide, monkeypatch):
    monkeypatch.setattr(FakeXM, "table", make_wide("2026-01-01", 60))
    result = runner.invoke(cli.app, ["ingest"])

    assert result.exit_code == 0, result.output
    assert "last_complete_day" in result.output
    assert "2026-03-01" in result.output
    assert sorted(path.name for path in data_dir.iterdir()) == ["processed", "raw"]


def test_ingest_fails_with_a_clear_message_when_xm_is_down(data_dir):
    result = runner.invoke(cli.app, ["ingest"])
    assert result.exit_code == 1
    assert "error: XM did not answer" in result.output
    assert list(data_dir.iterdir()) == []


def test_backtest_with_too_little_history_fails_cleanly(data_dir, make_wide, monkeypatch):
    monkeypatch.setattr(FakeXM, "table", make_wide("2026-06-01", 150))
    ingest(
        FakeXM(),
        today=date(2026, 11, 11),
        now=datetime(2026, 11, 12, 3, 0, tzinfo=UTC),
        data_dir=data_dir,
    )
    result = runner.invoke(cli.app, ["backtest"])
    assert result.exit_code == 1
    assert "error: no published history to forecast 2025-10-01" in result.output


@pytest.fixture
def registered_model(stored_history) -> str:
    """A small model fitted on the stored history and registered as the current one."""
    model = LightGBMForecaster(rounds=10, train_years=1)
    model.fit(
        History(
            demand=store.load_series(config.DEMAND.name, config.DATA_DIR),
            generation=store.load_series(config.GENERATION.name, config.DATA_DIR),
        )
    )
    entry = register_model(
        model,
        trained_at=datetime(2026, 11, 11, 16, 30, 5, tzinfo=UTC),
        data={"fingerprint": "0123456789abcdef"},
        backtest={
            "start": "2025-11-01",
            "end": "2026-10-31",
            "folds": 12,
            "metrics": {
                "seasonal-naive": {"mae": 340000.0, "mape": 3.4},
                "moving-average": {"mae": 300000.0, "mape": 3.0},
                "lightgbm": {"mae": 200000.0, "mape": 2.0},
            },
        },
        commit={"commit": "abc123", "dirty": False},
        models_dir=config.MODELS_DIR,
    )
    return entry["version"]


@pytest.fixture
def wednesday_morning(monkeypatch):
    now = pd.Timestamp("2026-11-12 08:00", tz=config.TIMEZONE)
    monkeypatch.setattr(cli, "bogota_now", lambda: now)


def test_forecast_prints_the_24_hours_of_tomorrow(registered_model, wednesday_morning):
    result = runner.invoke(cli.app, ["forecast"])

    assert result.exit_code == 0, result.output
    assert "Forecast of DemaCome for 2026-11-13" in result.output
    assert "issued 2026-11-12 08:00" in result.output
    assert f"model {registered_model}" in result.output
    assert "demand published up to 2026-11-06" in result.output
    assert "generation up to 2026-11-06" in result.output
    hours = [line for line in result.output.splitlines() if line[:5].endswith(":00")]
    assert [line[:5] for line in hours] == [f"{hour:02d}:00" for hour in range(24)]
    assert "GWh" in result.output


def test_forecast_without_a_model_says_how_to_get_it(stored_history, wednesday_morning):
    result = runner.invoke(cli.app, ["forecast"])
    assert result.exit_code == 1
    assert "run the `train` command first" in result.output


def test_forecast_without_data_says_how_to_get_it(data_dir, wednesday_morning):
    result = runner.invoke(cli.app, ["forecast"])
    assert result.exit_code == 1
    assert "error:" in result.output


@pytest.fixture
def served(monkeypatch) -> list[dict]:
    """Capture what `serve` hands to uvicorn instead of opening a port."""
    calls = []
    monkeypatch.setattr(
        cli.uvicorn, "run", lambda app, **kwargs: calls.append({"app": app} | kwargs)
    )
    return calls


def test_serve_starts_the_api_on_the_configured_address(registered_model, served, monkeypatch):
    monkeypatch.delenv("PORT", raising=False)

    result = runner.invoke(cli.app, ["serve"])

    assert result.exit_code == 0, result.output
    (call,) = served
    assert call["host"] == config.API_HOST
    assert call["port"] == config.API_PORT
    assert {route.path for route in call["app"].routes} >= {"/health", "/forecast", "/metrics"}


def test_serve_takes_the_port_from_the_environment(registered_model, served, monkeypatch):
    monkeypatch.setenv("PORT", "7860")

    result = runner.invoke(cli.app, ["serve", "--host", "0.0.0.0"])

    assert result.exit_code == 0, result.output
    assert (served[0]["host"], served[0]["port"]) == ("0.0.0.0", 7860)


def test_serve_without_a_model_does_not_start(stored_history, served):
    result = runner.invoke(cli.app, ["serve"])
    assert result.exit_code == 1
    assert "run the `train` command first" in result.output
    assert served == []


def at(text: str, monkeypatch) -> None:
    now = pd.Timestamp(text, tz=config.TIMEZONE)
    monkeypatch.setattr(cli, "bogota_now", lambda: now)


def test_daily_ingests_and_logs_the_forecast_for_tomorrow(registered_model, monkeypatch):
    at("2026-11-10 15:17", monkeypatch)

    result = runner.invoke(cli.app, ["daily"])

    assert result.exit_code == 0, result.output
    assert "last_complete_day" in result.output
    assert "Evaluated days: 0" in result.output
    assert "Logged the forecast for 2026-11-11" in result.output
    assert f"model {registered_model}" in result.output
    log = read_log(config.DATA_DIR)
    assert len(log) == 24
    assert set(log["target_day"]) == {pd.Timestamp("2026-11-11", tz=config.TIMEZONE)}


def test_daily_twice_on_the_same_day_logs_once(registered_model, monkeypatch):
    at("2026-11-10 15:17", monkeypatch)
    runner.invoke(cli.app, ["daily"])
    at("2026-11-10 20:00", monkeypatch)

    result = runner.invoke(cli.app, ["daily"])

    assert result.exit_code == 0, result.output
    assert "The forecast for 2026-11-11 was already logged at 2026-11-10 15:17" in result.output
    assert len(read_log(config.DATA_DIR)) == 24


def test_daily_fails_with_a_clear_message_when_xm_is_down(registered_model, monkeypatch):
    at("2026-11-10 15:17", monkeypatch)
    monkeypatch.setattr(FakeXM, "table", None)

    result = runner.invoke(cli.app, ["daily"])

    assert result.exit_code == 1
    assert "error: XM did not answer" in result.output
    assert read_log(config.DATA_DIR).empty


def test_daily_refuses_to_forecast_with_old_data(registered_model, monkeypatch):
    at("2026-11-25 15:17", monkeypatch)

    result = runner.invoke(cli.app, ["daily"])

    assert result.exit_code == 1
    assert "error: the demand is published up to 2026-11-06" in result.output
    assert read_log(config.DATA_DIR).empty


def evaluated_days(mape: float, days: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_day": pd.date_range("2026-11-01", periods=days, freq="D", tz=config.TIMEZONE),
            "day_type": "workday",
            "model_version": "20261111T163005Z",
            "actual_kwh": 2.4e8,
            "predicted_kwh": 2.4e8,
            "mae_kwh": 2e5,
            "mape_percent": mape,
        }
    )


def test_alert_writes_the_warning_when_the_recent_error_is_too_high(registered_model, tmp_path):
    write_metrics(evaluated_days(mape=3.5, days=7), config.DATA_DIR)
    output = tmp_path / "alert.md"

    result = runner.invoke(cli.app, ["alert", "--output", str(output)])

    assert result.exit_code == 0, result.output
    assert "3.50 %" in output.read_text(encoding="utf-8")
    assert "above the alert threshold of 3.00 %" in result.output


def test_alert_writes_nothing_when_the_error_is_normal(registered_model, tmp_path):
    write_metrics(evaluated_days(mape=2.1, days=7), config.DATA_DIR)
    output = tmp_path / "alert.md"

    result = runner.invoke(cli.app, ["alert", "--output", str(output)])

    assert result.exit_code == 0, result.output
    assert "No alert" in result.output
    assert "2.10 %" in result.output
    assert not output.exists()


def test_alert_waits_for_seven_evaluated_days(registered_model, tmp_path):
    write_metrics(evaluated_days(mape=9.0, days=6), config.DATA_DIR)
    output = tmp_path / "alert.md"

    result = runner.invoke(cli.app, ["alert", "--output", str(output)])

    assert result.exit_code == 0, result.output
    assert "No alert: only 6 evaluated days" in result.output
    assert not output.exists()


def test_report_builds_the_panel(registered_model, monkeypatch, tmp_path):
    at("2026-11-10 15:17", monkeypatch)
    runner.invoke(cli.app, ["daily"])
    site_dir = tmp_path / "site"

    result = runner.invoke(cli.app, ["report", "--site-dir", str(site_dir)])

    assert result.exit_code == 0, result.output
    assert "index.html" in result.output
    page = (site_dir / "index.html").read_text(encoding="utf-8")
    assert "2026-11-11T00:00:00-05:00" in page


def test_report_without_a_model_fails_cleanly(stored_history, tmp_path):
    result = runner.invoke(cli.app, ["report", "--site-dir", str(tmp_path / "site")])
    assert result.exit_code == 1
    assert "run the `train` command first" in result.output


def test_drift_has_nothing_to_report_before_a_week_of_forecasts(registered_model, tmp_path):
    result = runner.invoke(cli.app, ["drift", "--site-dir", str(tmp_path / "site")])

    assert result.exit_code == 0, result.output
    assert (
        f"No drift report: it needs {config.DRIFT_MIN_DAYS} logged forecast days" in result.output
    )
    assert not (tmp_path / "site").exists()


def test_drift_says_which_features_drifted(registered_model, monkeypatch, tmp_path):
    summary = DriftSummary(
        model_version=registered_model,
        reference_start="2025-08-04",
        reference_end="2026-11-06",
        current_start="2026-11-11",
        current_end="2026-11-20",
        current_days=10,
        method="Wasserstein distance (normed)",
        quantile=0.99,
        thresholds={"week_level": 0.25, "profile_shape": 0.1},
        scores={"week_level": 0.31, "profile_shape": 0.02},
        drifted=["week_level"],
    )
    seen = []
    monkeypatch.setattr(cli, "write_drift", lambda *args: seen.append(args) or summary)
    site_dir = tmp_path / "site"

    result = runner.invoke(cli.app, ["drift", "--site-dir", str(site_dir)])

    assert result.exit_code == 0, result.output
    assert "Drift over 10 logged days (2026-11-11 to 2026-11-20)" in result.output
    assert "1 of 2 features drifted (week_level)" in result.output
    assert seen == [(config.DATA_DIR, config.MODELS_DIR, site_dir)]


def test_drift_asks_for_the_monitoring_group_when_evidently_is_missing(
    registered_model, monkeypatch, tmp_path
):
    def missing(*args):
        raise ImportError("No module named 'evidently'")

    monkeypatch.setattr(cli, "write_drift", missing)

    result = runner.invoke(cli.app, ["drift", "--site-dir", str(tmp_path / "site")])

    assert result.exit_code == 1
    assert "uv sync --group monitoring" in result.output


def test_retrain_keeps_the_model_until_enough_days_are_published(registered_model):
    result = runner.invoke(cli.app, ["retrain"])

    assert result.exit_code == 0, result.output
    assert f"Kept model {registered_model}" in result.output
    assert f"of the {config.RETRAIN_MIN_DAYS} days needed" in result.output
    registry = read_registry(config.MODELS_DIR)
    assert registry["current"] == registered_model
    assert len(registry["models"]) == 1
    (decision,) = registry["decisions"]
    assert decision["promoted"] is False
    assert decision["current_version"] == registered_model
    assert decision["days"] == 0
    assert decision["new_version"] is None


def test_retrain_promotes_when_retraining_beat_the_current_model(
    registered_model, two_month_windows, monkeypatch
):
    comparison = Comparison(
        start=pd.Timestamp("2026-10-01", tz=config.TIMEZONE),
        end=pd.Timestamp("2026-11-06", tz=config.TIMEZONE),
        days=37,
        champion_mape=2.4,
        challenger_mape=2.1,
    )
    monkeypatch.setattr(cli, "compare", lambda *args, **kwargs: comparison)

    result = runner.invoke(cli.app, ["retrain"])

    assert result.exit_code == 0, result.output
    assert "Promoted a retrained model" in result.output
    assert "MAPE 2.40 % for the current model and 2.10 %" in result.output
    registry = read_registry(config.MODELS_DIR)
    assert registry["current"] != registered_model
    assert len(registry["models"]) == 2
    (decision,) = registry["decisions"]
    assert decision["promoted"] is True
    assert decision["new_version"] == registry["current"]
    assert decision["current_mape"] == 2.4
    assert decision["retrained_mape"] == 2.1
    assert decision["window_start"] == "2026-10-01"
    assert decision["window_end"] == "2026-11-06"


def test_retrain_records_that_the_current_model_was_kept(registered_model, monkeypatch):
    comparison = Comparison(
        start=pd.Timestamp("2026-10-01", tz=config.TIMEZONE),
        end=pd.Timestamp("2026-11-06", tz=config.TIMEZONE),
        days=37,
        champion_mape=2.0,
        challenger_mape=2.1,
    )
    monkeypatch.setattr(cli, "compare", lambda *args, **kwargs: comparison)

    result = runner.invoke(cli.app, ["retrain"])

    assert result.exit_code == 0, result.output
    assert f"Kept model {registered_model}: the current model was at least as good" in result.output
    registry = read_registry(config.MODELS_DIR)
    assert registry["current"] == registered_model
    assert registry["decisions"][0]["promoted"] is False


def test_retrain_without_a_model_fails_cleanly(stored_history):
    result = runner.invoke(cli.app, ["retrain"])
    assert result.exit_code == 1
    assert "run the `train` command first" in result.output
