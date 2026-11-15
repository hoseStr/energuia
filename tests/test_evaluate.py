import numpy as np
import pandas as pd
import pytest

from demand_forecast import config
from demand_forecast.monitoring.evaluate import (
    alert_message,
    evaluate,
    read_metrics,
    recent_mape,
    write_metrics,
)
from demand_forecast.serving.forecast import Forecast
from demand_forecast.serving.log import append_forecast, read_log

BOGOTA = config.TIMEZONE
VERSION = "20261113T215929Z"
ONE_DAY = pd.Timedelta(days=1)


def day(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=BOGOTA)


@pytest.fixture
def demand() -> pd.Series:
    """Actual demand published up to 2026-11-20: 1e7 kWh in every hour, 5 % more at noon."""
    index = pd.date_range(day("2026-10-01"), day("2026-11-20 23:00"), freq="h")
    return pd.Series(np.where(index.hour == 12, 1.05e7, 1e7), index=index)


def log_forecasts(data_dir, demand: pd.Series, errors: dict[str, float]) -> pd.DataFrame:
    """Log, for each day, a forecast that is off by the given fraction of 1e7 kWh in every hour."""
    for text, error in errors.items():
        target_day = day(text)
        index = pd.date_range(target_day, periods=24, freq="h")
        actual = demand.reindex(index).fillna(1e7)
        append_forecast(
            Forecast(
                issued_at=target_day - ONE_DAY + pd.Timedelta(hours=15),
                target_day=target_day,
                model_version=VERSION,
                demand_last_day=target_day - 7 * ONE_DAY,
                generation_last_day=target_day - 5 * ONE_DAY,
                values=actual + error * 1e7,
            ),
            data_dir,
        )
    return read_log(data_dir)


def test_a_published_day_gets_its_error(tmp_path, demand):
    log = log_forecasts(tmp_path, demand, {"2026-11-13": 0.02})

    (row,) = evaluate(log, demand).to_dict(orient="records")

    assert row["target_day"] == day("2026-11-13")
    assert row["day_type"] == "workday"
    assert row["model_version"] == VERSION
    assert row["actual_kwh"] == pytest.approx(23 * 1e7 + 1.05e7)
    assert row["predicted_kwh"] == pytest.approx(23 * 1e7 + 1.05e7 + 24 * 2e5)
    assert row["mae_kwh"] == pytest.approx(2e5)
    assert row["mape_percent"] == pytest.approx((23 * 2.0 + 2 / 1.05) / 24)


def test_a_forecast_that_was_too_low_counts_like_one_that_was_too_high(tmp_path, demand):
    log = log_forecasts(tmp_path, demand, {"2026-11-13": -0.02})
    (row,) = evaluate(log, demand).to_dict(orient="records")
    assert row["mae_kwh"] == pytest.approx(2e5)
    assert row["predicted_kwh"] < row["actual_kwh"]


def test_days_whose_demand_is_not_published_yet_are_left_out(tmp_path, demand):
    log = log_forecasts(tmp_path, demand, {"2026-11-20": 0.01, "2026-11-21": 0.01})

    daily = evaluate(log, demand)

    assert list(daily["target_day"]) == [day("2026-11-20")]


def test_a_holiday_is_labelled_as_such(tmp_path, demand):
    log = log_forecasts(tmp_path, demand, {"2026-11-14": 0.01, "2026-11-16": 0.01})
    assert list(evaluate(log, demand)["day_type"]) == ["saturday", "sunday_or_holiday"]


def test_an_empty_log_gives_an_empty_table(tmp_path, demand):
    daily = evaluate(read_log(tmp_path), demand)
    assert daily.empty
    assert list(daily.columns) == [
        "target_day",
        "day_type",
        "model_version",
        "actual_kwh",
        "predicted_kwh",
        "mae_kwh",
        "mape_percent",
    ]


def test_the_metrics_file_round_trips_and_is_stable(tmp_path, demand):
    log = log_forecasts(tmp_path, demand, {"2026-11-13": 0.02, "2026-11-14": 0.03})
    daily = evaluate(log, demand)

    path = write_metrics(daily, tmp_path)
    first = path.read_bytes()
    write_metrics(evaluate(log, demand), tmp_path)

    assert path == tmp_path / "metrics" / "daily.csv"
    assert path.read_bytes() == first
    assert b"\r" not in first
    assert first.decode().splitlines()[1].startswith(f"2026-11-13,workday,{VERSION},")
    stored = read_metrics(tmp_path)
    assert list(stored["target_day"]) == list(daily["target_day"])
    assert stored["mape_percent"].to_numpy() == pytest.approx(
        daily["mape_percent"].to_numpy(), abs=1e-4
    )
    assert list(stored["model_version"]) == [VERSION, VERSION]


def test_no_metrics_file_reads_as_an_empty_table(tmp_path):
    assert read_metrics(tmp_path).empty


def daily_with(mapes: list[float]) -> pd.DataFrame:
    days = pd.date_range(day("2026-11-01"), periods=len(mapes), freq="D")
    return pd.DataFrame({"target_day": days, "mape_percent": mapes})


def test_the_recent_error_is_the_mean_of_the_last_seven_evaluated_days():
    daily = daily_with([9.0, 9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert recent_mape(daily) == pytest.approx(4.0)


def test_there_is_no_recent_error_with_fewer_than_seven_days():
    assert recent_mape(daily_with([2.0] * 6)) is None
    assert recent_mape(daily_with([])) is None


def test_the_alert_fires_above_one_and_a_half_times_the_backtest_error():
    message = alert_message(daily_with([3.1] * 7), backtest_mape=2.0)

    assert "3.10 %" in message
    assert "3.00 %" in message
    assert "2026-11-01" in message
    assert "2026-11-07" in message


def test_no_alert_at_or_below_the_threshold_or_without_enough_days():
    assert alert_message(daily_with([3.0] * 7), backtest_mape=2.0) is None
    assert alert_message(daily_with([1.5] * 30), backtest_mape=2.0) is None
    assert alert_message(daily_with([9.0] * 6), backtest_mape=2.0) is None
