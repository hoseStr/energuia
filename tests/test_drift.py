import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from demand_forecast import config
from demand_forecast.data import store
from demand_forecast.features.build import training_frame
from demand_forecast.models.backtest import History
from demand_forecast.models.registry import register_model
from demand_forecast.models.train import LightGBMForecaster
from demand_forecast.monitoring.drift import (
    DRIFT_FEATURES,
    current_features,
    drift_scores,
    read_drift,
    reference_features,
    window_thresholds,
    write_drift,
)
from demand_forecast.serving.forecast import Forecast
from demand_forecast.serving.log import append_forecast, read_log

BOGOTA = config.TIMEZONE
ONE_DAY = pd.Timedelta(days=1)


def day(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=BOGOTA)


@pytest.fixture(scope="module")
def history() -> History:
    """Noisy synthetic demand and generation from 2026-01-01 to 2026-11-20."""
    rng = np.random.default_rng(7)
    index = pd.date_range(day("2026-01-01"), day("2026-11-20 23:00"), freq="h")
    shape = 1 + 0.2 * np.sin(index.hour / 24 * 2 * np.pi)
    clean = pd.Series(1e7 * shape * np.where(index.dayofweek == 6, 0.85, 1.0), index=index)
    return History(
        demand=clean * rng.normal(1, 0.03, len(clean)),
        generation=clean * rng.normal(1, 0.03, len(clean)),
    )


def log_days(data_dir, targets: list[str]) -> pd.DataFrame:
    """Log one forecast per day, each made with demand up to 6 days and generation 4 days before."""
    for text in targets:
        target_day = day(text)
        append_forecast(
            Forecast(
                issued_at=target_day - ONE_DAY + pd.Timedelta(hours=15),
                target_day=target_day,
                model_version="20261113T163005Z",
                demand_last_day=target_day - 6 * ONE_DAY,
                generation_last_day=target_day - 4 * ONE_DAY,
                values=pd.Series(1e7, index=pd.date_range(target_day, periods=24, freq="h")),
            ),
            data_dir,
        )
    return read_log(data_dir)


ENTRY = {
    "version": "20261113T163005Z",
    "data": {"first_target_day": "2026-03-01", "last_target_day": "2026-10-31"},
    "model": {"horizon_days": 1, "lag_days": 5, "generation_lag_days": 3},
}


def test_the_current_features_are_those_of_the_logged_days(tmp_path, history):
    log = log_days(tmp_path, ["2026-11-15", "2026-11-16", "2026-11-17"])

    current = current_features(history, log)

    assert list(current.columns) == DRIFT_FEATURES
    assert len(current) == 3 * 24
    assert current.index[0] == day("2026-11-15")
    assert current.index[-1] == day("2026-11-17 23:00")
    assert current.notna().all().all()


def test_the_current_features_only_use_what_each_forecast_could_see(tmp_path, history):
    log = log_days(tmp_path, ["2026-11-15", "2026-11-16"])
    tampered = History(demand=history.demand.copy(), generation=history.generation.copy())
    tampered.demand[tampered.demand.index >= day("2026-11-11")] *= 3
    tampered.generation[tampered.generation.index >= day("2026-11-13")] *= 3

    honest = current_features(history, log)
    blind = current_features(tampered, log)

    pd.testing.assert_frame_equal(honest, blind)


def test_the_cuts_come_from_the_log_and_not_from_the_configured_lags(tmp_path, history):
    usual = log_days(tmp_path / "usual", ["2026-11-15"])
    late = usual.assign(demand_last_day=usual["demand_last_day"] - 3 * ONE_DAY)

    on_time = current_features(history, usual)
    stale = current_features(history, late)

    assert not on_time["demand_last_day_dev"].equals(stale["demand_last_day_dev"])


def test_only_the_last_days_of_the_log_count(tmp_path, history):
    log = log_days(tmp_path, ["2026-11-14", "2026-11-15", "2026-11-16", "2026-11-17"])

    current = current_features(history, log, days=2)

    assert len(current) == 2 * 24
    assert current.index[0] == day("2026-11-16")


def test_an_empty_log_has_no_current_features(tmp_path, history):
    assert current_features(history, read_log(tmp_path)).empty


def test_the_reference_is_the_training_period_of_the_registered_model(history):
    reference = reference_features(history, ENTRY)

    assert list(reference.columns) == DRIFT_FEATURES
    assert reference.index[0] == day("2026-03-01")
    assert reference.index[-1] == day("2026-10-31 23:00")
    trained_on = History(
        demand=history.demand[history.demand.index < day("2026-11-01")],
        generation=history.generation,
    )
    expected = training_frame(trained_on, first_target=day("2026-03-01"))[DRIFT_FEATURES]
    pd.testing.assert_frame_equal(reference, expected)


@pytest.fixture
def stored(tmp_path, history) -> tuple:
    """The synthetic history on disk, with a model registered on it."""
    pytest.importorskip("evidently")
    data_dir, models_dir = tmp_path / "data", tmp_path / "models"
    for name, series in (
        (config.DEMAND.name, history.demand),
        (config.GENERATION.name, history.generation),
    ):
        table = pd.DataFrame(
            {"timestamp": series.index, "value": series.to_numpy(), "anomaly": False}
        )
        store.write_table(table, store.processed_dir(name, data_dir), "timestamp")
    model = LightGBMForecaster(rounds=5, train_years=1)
    model.fit(
        History(
            demand=history.demand[history.demand.index < day("2026-11-01")],
            generation=history.generation,
        )
    )
    register_model(
        model,
        trained_at=datetime(2026, 11, 13, 16, 30, 5, tzinfo=UTC),
        data={"fingerprint": "0123456789abcdef"},
        backtest={"start": "2025-11-01", "end": "2026-10-31", "folds": 12, "metrics": {}},
        commit={"commit": "abc123", "dirty": False},
        models_dir=models_dir,
    )
    return data_dir, models_dir, tmp_path / "site"


def test_the_report_and_its_summary_are_written_to_the_site(stored):
    data_dir, models_dir, site_dir = stored
    log_days(data_dir, [f"2026-11-{number:02d}" for number in range(10, 21)])

    summary = write_drift(data_dir, models_dir, site_dir)

    assert (site_dir / "drift.html").stat().st_size > 10_000
    assert summary.current_days == 11
    assert summary.current_start == "2026-11-10"
    assert summary.current_end == "2026-11-20"
    assert summary.reference_end == "2026-10-31"
    assert summary.quantile == config.DRIFT_QUANTILE
    assert set(summary.thresholds) == set(DRIFT_FEATURES)
    assert set(summary.scores) == set(DRIFT_FEATURES)
    assert all(score >= 0 for score in summary.scores.values())
    assert set(summary.drifted) <= set(DRIFT_FEATURES)
    assert read_drift(site_dir) == summary
    stored_summary = json.loads((site_dir / "drift.json").read_text(encoding="utf-8"))
    assert stored_summary["model_version"] == "20261113T163005Z"


def test_a_shift_in_the_demand_shows_up_as_drift(stored, history):
    data_dir, models_dir, site_dir = stored
    shifted = history.demand.copy()
    shifted[shifted.index >= day("2026-11-01")] *= 1.25
    table = pd.DataFrame(
        {"timestamp": shifted.index, "value": shifted.to_numpy(), "anomaly": False}
    )
    store.write_table(table, store.processed_dir(config.DEMAND.name, data_dir), "timestamp")
    log_days(data_dir, [f"2026-11-{number:02d}" for number in range(10, 21)])

    summary = write_drift(data_dir, models_dir, site_dir)

    assert "week_level" in summary.drifted or "demand_last_day_dev" in summary.drifted
    assert "generation_last_day_dev" in summary.drifted


def test_without_logged_forecasts_there_is_no_report(stored):
    data_dir, models_dir, site_dir = stored

    assert write_drift(data_dir, models_dir, site_dir) is None
    assert read_drift(site_dir) is None
    assert not site_dir.exists()


def test_the_thresholds_are_what_ordinary_windows_of_the_reference_rarely_reach(history):
    pytest.importorskip("scipy")
    reference = reference_features(history, ENTRY)

    loose = window_thresholds(reference, 7 * 24, quantile=0.5)
    strict = window_thresholds(reference, 7 * 24, quantile=0.99)
    longer = window_thresholds(reference, 28 * 24, quantile=0.99)

    assert set(strict) == set(DRIFT_FEATURES)
    assert all(0 < loose[name] < strict[name] for name in DRIFT_FEATURES)
    assert sum(longer[name] < strict[name] for name in DRIFT_FEATURES) >= 8


def test_a_reference_shorter_than_the_window_is_an_error(history):
    pytest.importorskip("scipy")
    reference = reference_features(history, ENTRY).iloc[: 5 * 24]
    with pytest.raises(ValueError, match="shorter than"):
        window_thresholds(reference, 7 * 24)


def test_the_thresholds_use_the_same_score_as_the_report(tmp_path, history):
    pytest.importorskip("evidently")
    from scipy.stats import wasserstein_distance

    reference = reference_features(history, ENTRY)
    current = reference.iloc[-14 * 24 :] * 1.02
    thresholds = dict.fromkeys(DRIFT_FEATURES, 0.5)

    scores = drift_scores(reference, current, tmp_path / "drift.html", thresholds)

    for name in DRIFT_FEATURES:
        values = reference[name].to_numpy()
        expected = wasserstein_distance(values, current[name].to_numpy()) / values.std()
        assert scores[name] == pytest.approx(expected, rel=1e-6)


def test_an_ordinary_stretch_of_the_training_period_does_not_drift(stored):
    data_dir, models_dir, site_dir = stored
    log_days(data_dir, [f"2026-10-{number:02d}" for number in range(4, 32)])

    summary = write_drift(data_dir, models_dir, site_dir)

    assert summary.current_days == 28
    assert len(summary.drifted) <= 1
