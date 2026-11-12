import numpy as np
import pandas as pd
import pandera.errors
import pytest

from demand_forecast import config
from demand_forecast.data.schema import FETCHED_AT_DTYPE, validate_hourly, validate_raw
from demand_forecast.data.transform import hourly_table


@pytest.fixture
def hourly(make_wide) -> pd.DataFrame:
    return hourly_table(make_wide("2026-01-01", 40))


@pytest.fixture
def raw(make_wide) -> pd.DataFrame:
    fetched_at = pd.Timestamp("2026-11-10 22:46", tz="UTC")
    frame = make_wide("2026-01-01", 40).assign(fetched_at=fetched_at)
    return frame.astype({"fetched_at": FETCHED_AT_DTYPE})


def drop_rows(frame: pd.DataFrame, rows: list[int]) -> pd.DataFrame:
    return frame.drop(index=rows).reset_index(drop=True)


class TestHourlySchema:
    def test_a_clean_table_is_valid(self, hourly):
        pd.testing.assert_frame_equal(validate_hourly(hourly), hourly)

    def test_duplicated_timestamps_are_rejected(self, hourly):
        duplicated = pd.concat([hourly, hourly.iloc[[5]]]).sort_values(
            "timestamp", ignore_index=True
        )
        with pytest.raises(pandera.errors.SchemaErrors):
            validate_hourly(duplicated)

    def test_a_missing_hour_is_rejected(self, hourly):
        with pytest.raises(pandera.errors.SchemaErrors, match="consecutive hours"):
            validate_hourly(drop_rows(hourly, [100]))

    def test_a_missing_day_is_rejected(self, hourly):
        with pytest.raises(pandera.errors.SchemaErrors, match="consecutive hours"):
            validate_hourly(drop_rows(hourly, list(range(48, 72))))

    def test_a_last_day_cut_short_is_rejected(self, hourly):
        with pytest.raises(pandera.errors.SchemaErrors, match="whole days"):
            validate_hourly(hourly.iloc[:-1])

    @pytest.mark.parametrize(
        "value",
        [0.0, -1.0, np.nan, config.PLAUSIBLE_KWH_MIN / 2, config.PLAUSIBLE_KWH_MAX * 2],
        ids=["zero", "negative", "nan", "too-small", "too-large"],
    )
    def test_implausible_values_are_rejected(self, hourly, value):
        hourly.loc[10, "value"] = value
        with pytest.raises(pandera.errors.SchemaErrors):
            validate_hourly(hourly)

    def test_other_timezones_are_rejected(self, hourly):
        hourly["timestamp"] = hourly["timestamp"].dt.tz_convert("UTC")
        with pytest.raises(pandera.errors.SchemaErrors):
            validate_hourly(hourly)

    def test_unexpected_columns_are_rejected(self, hourly):
        with pytest.raises(pandera.errors.SchemaErrors):
            validate_hourly(hourly.assign(extra=1))


class TestRawSchema:
    def test_partial_days_with_missing_hours_are_valid(self, raw):
        raw.loc[39, ["hour_23", "hour_24"]] = np.nan
        pd.testing.assert_frame_equal(validate_raw(raw), raw)

    def test_duplicated_dates_are_rejected(self, raw):
        with pytest.raises(pandera.errors.SchemaErrors):
            validate_raw(pd.concat([raw, raw.iloc[[-1]]], ignore_index=True))

    def test_unsorted_dates_are_rejected(self, raw):
        with pytest.raises(pandera.errors.SchemaErrors, match="sorted"):
            validate_raw(raw.iloc[::-1].reset_index(drop=True))

    def test_negative_values_are_rejected(self, raw):
        raw.loc[3, "hour_01"] = -5.0
        with pytest.raises(pandera.errors.SchemaErrors):
            validate_raw(raw)
