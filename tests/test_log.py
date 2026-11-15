import numpy as np
import pandas as pd
import pytest

from demand_forecast import config
from demand_forecast.serving.forecast import Forecast
from demand_forecast.serving.log import (
    LogError,
    append_forecast,
    latest_forecast,
    predictions_dir,
    read_log,
)

BOGOTA = config.TIMEZONE
VERSION = "20261113T215929Z"


def day(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=BOGOTA)


def make_forecast(
    target: str = "2026-11-15",
    issued: str = "2026-11-14 15:17:42",
    *,
    version: str = VERSION,
    level: float = 9e6,
    hours: int = 24,
) -> Forecast:
    target_day = day(target)
    index = pd.date_range(target_day, periods=hours, freq="h")
    return Forecast(
        issued_at=day(issued),
        target_day=target_day,
        model_version=version,
        demand_last_day=target_day - pd.Timedelta(days=7),
        generation_last_day=target_day - pd.Timedelta(days=5),
        values=pd.Series(level + np.arange(hours) * 1000.123456, index=index),
    )


def test_the_first_forecast_creates_the_file_of_its_year(tmp_path):
    path = append_forecast(make_forecast(), tmp_path)

    assert path == tmp_path / "predictions" / "2026.csv"
    lines = path.read_text(encoding="utf-8").split("\n")
    assert lines[0] == (
        "timestamp,predicted_kwh,issued_at,model_version,demand_last_day,generation_last_day"
    )
    assert lines[1] == (
        "2026-11-15T00:00:00-05:00,9000000.000,2026-11-14T15:17:42-05:00,"
        f"{VERSION},2026-11-08,2026-11-10"
    )
    assert lines[24].startswith("2026-11-15T23:00:00-05:00,9023002.839,")
    assert lines[25:] == [""]


def test_the_file_uses_unix_line_endings_on_every_platform(tmp_path):
    path = append_forecast(make_forecast(), tmp_path)
    assert b"\r" not in path.read_bytes()


def test_reading_gives_back_what_was_logged(tmp_path):
    forecast = make_forecast()
    append_forecast(forecast, tmp_path)

    log = read_log(tmp_path)

    assert len(log) == 24
    assert list(log["timestamp"]) == list(forecast.values.index)
    assert str(log["timestamp"].dt.tz) == BOGOTA
    assert log["predicted_kwh"].to_numpy() == pytest.approx(forecast.values.to_numpy(), abs=1e-3)
    assert set(log["target_day"]) == {day("2026-11-15")}
    assert set(log["issued_at"]) == {day("2026-11-14 15:17:42")}
    assert set(log["model_version"]) == {VERSION}
    assert set(log["demand_last_day"]) == {day("2026-11-08")}
    assert set(log["generation_last_day"]) == {day("2026-11-10")}


def test_an_empty_log_reads_as_an_empty_table(tmp_path):
    log = read_log(tmp_path)

    assert log.empty
    assert "target_day" in log.columns
    assert latest_forecast(log) is None


def test_a_new_forecast_only_adds_lines(tmp_path):
    path = append_forecast(make_forecast("2026-11-15", "2026-11-14 15:17"), tmp_path)
    before = path.read_bytes()

    append_forecast(make_forecast("2026-11-16", "2026-11-15 15:20", level=1e7), tmp_path)

    after = path.read_bytes()
    assert after.startswith(before)
    assert len(after.split(b"\n")) == len(before.split(b"\n")) + 24
    assert list(read_log(tmp_path)["target_day"].unique()) == [day("2026-11-15"), day("2026-11-16")]


def test_a_day_that_was_already_forecast_cannot_be_forecast_again(tmp_path):
    path = append_forecast(make_forecast(), tmp_path)
    before = path.read_bytes()

    with pytest.raises(LogError, match="2026-11-15 is already in the log"):
        append_forecast(make_forecast(issued="2026-11-14 20:00", level=1e7), tmp_path)

    assert path.read_bytes() == before


def test_a_day_before_the_last_logged_one_cannot_be_filled_in_later(tmp_path):
    path = append_forecast(make_forecast("2026-11-17", "2026-11-16 15:17"), tmp_path)
    before = path.read_bytes()

    with pytest.raises(LogError, match="the log already reaches 2026-11-17"):
        append_forecast(make_forecast("2026-11-16", "2026-11-15 15:17"), tmp_path)

    assert path.read_bytes() == before


@pytest.mark.parametrize("issued", ["2026-11-15 00:00", "2026-11-15 08:30", "2026-11-20 10:00"])
def test_a_forecast_issued_once_its_day_started_is_not_a_forecast(tmp_path, issued):
    with pytest.raises(LogError, match="was issued after 2026-11-15 started"):
        append_forecast(make_forecast("2026-11-15", issued), tmp_path)

    assert not predictions_dir(tmp_path).exists()


def test_a_forecast_without_its_24_hours_is_rejected(tmp_path):
    with pytest.raises(LogError, match="24 hours"):
        append_forecast(make_forecast(hours=23), tmp_path)


def test_a_forecast_with_missing_values_is_rejected(tmp_path):
    forecast = make_forecast()
    forecast.values.iloc[5] = np.nan

    with pytest.raises(LogError, match="positive"):
        append_forecast(forecast, tmp_path)


def test_each_year_has_its_own_file_and_the_old_one_is_left_alone(tmp_path):
    old = append_forecast(make_forecast("2026-12-31", "2026-12-30 15:17"), tmp_path)
    before = old.read_bytes()

    new = append_forecast(make_forecast("2027-01-01", "2026-12-31 15:17"), tmp_path)

    assert new == tmp_path / "predictions" / "2027.csv"
    assert old.read_bytes() == before
    log = read_log(tmp_path)
    assert list(log["target_day"].unique()) == [day("2026-12-31"), day("2027-01-01")]
    assert log["timestamp"].is_monotonic_increasing


def test_the_latest_forecast_is_the_last_one_logged(tmp_path):
    append_forecast(make_forecast("2026-11-15", "2026-11-14 15:17"), tmp_path)
    last = make_forecast("2026-11-16", "2026-11-15 15:20:05", version="20261120T160000Z", level=1e7)
    append_forecast(last, tmp_path)

    found = latest_forecast(read_log(tmp_path))

    assert found.target_day == last.target_day
    assert found.issued_at == last.issued_at
    assert found.model_version == "20261120T160000Z"
    assert found.demand_last_day == last.demand_last_day
    assert found.generation_last_day == last.generation_last_day
    assert list(found.values.index) == list(last.values.index)
    assert found.values.to_numpy() == pytest.approx(last.values.to_numpy(), abs=1e-3)
