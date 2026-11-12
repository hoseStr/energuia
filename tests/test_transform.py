import numpy as np
import pandas as pd

from demand_forecast.data.schema import HOUR_COLUMNS, TIMESTAMP_DTYPE
from demand_forecast.data.transform import (
    anomalous_hours,
    daily_totals,
    hourly_table,
    wide_to_long,
)


class TestWideToLong:
    def test_hour_nn_starts_at_nn_minus_one_local_time(self, demand_wide):
        long = wide_to_long(demand_wide.iloc[[0]])
        first, last = long.iloc[0], long.iloc[-1]
        assert first["timestamp"] == pd.Timestamp("2026-10-01 00:00", tz="America/Bogota")
        assert first["value"] == demand_wide.loc[0, "hour_01"]
        assert last["timestamp"] == pd.Timestamp("2026-10-01 23:00", tz="America/Bogota")
        assert last["value"] == demand_wide.loc[0, "hour_24"]

    def test_timestamps_are_bogota_time_without_dst(self, demand_wide):
        long = wide_to_long(demand_wide)
        assert long["timestamp"].dtype == TIMESTAMP_DTYPE
        assert {ts.utcoffset() for ts in long["timestamp"]} == {pd.Timedelta(hours=-5)}

    def test_one_row_per_hour_in_order(self, demand_wide):
        long = wide_to_long(demand_wide)
        assert list(long.columns) == ["timestamp", "value"]
        assert len(long) == 24 * len(demand_wide)
        assert long["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all()

    def test_values_keep_their_day_and_hour(self, demand_wide):
        long = wide_to_long(demand_wide).set_index("timestamp")["value"]
        for _, row in demand_wide.iterrows():
            day = long[long.index.normalize() == row["date"].tz_localize("America/Bogota")]
            assert list(day) == list(row[HOUR_COLUMNS])

    def test_input_order_does_not_matter(self, demand_wide):
        shuffled = demand_wide.sample(frac=1, random_state=0)
        pd.testing.assert_frame_equal(wide_to_long(shuffled), wide_to_long(demand_wide))


class TestDailyTotals:
    def test_sums_the_24_hours(self, demand_wide):
        totals = daily_totals(demand_wide)
        assert totals.loc["2026-11-05"] == demand_wide.iloc[-3][HOUR_COLUMNS].sum()
        assert round(totals.loc["2026-11-05"] / 1e6, 2) == 246.13

    def test_a_missing_hour_makes_the_total_nan(self, demand_wide):
        demand_wide.loc[3, "hour_10"] = np.nan
        totals = daily_totals(demand_wide)
        assert np.isnan(totals.iloc[3])
        assert totals.drop(totals.index[3]).notna().all()


class TestAnomalousHours:
    def test_a_regular_series_has_no_anomalies(self, make_wide):
        assert not anomalous_hours(make_wide("2026-01-01", 120)).to_numpy().any()

    def test_real_sundays_and_holidays_are_not_anomalies(self, demand_wide):
        complete = demand_wide.iloc[:-2]
        assert not anomalous_hours(complete).to_numpy().any()

    def test_a_blackout_hour_is_flagged_and_nothing_else(self, make_wide):
        wide = make_wide("2026-01-01", 120)
        wide.loc[60, "hour_12"] *= 0.3
        flags = anomalous_hours(wide)
        assert flags.loc[wide.loc[60, "date"], "hour_12"]
        assert flags.to_numpy().sum() == 1

    def test_flags_only_depend_on_earlier_days(self, make_wide):
        wide = make_wide("2026-01-01", 120)
        wide.loc[60, "hour_12"] *= 0.3
        altered = wide.copy()
        altered.loc[61:, HOUR_COLUMNS] *= 5
        pd.testing.assert_frame_equal(
            anomalous_hours(wide).iloc[:61], anomalous_hours(altered).iloc[:61]
        )

    def test_days_without_a_full_reference_are_never_flagged(self, make_wide):
        wide = make_wide("2026-01-01", 40)
        wide.loc[10, "hour_05"] *= 0.1
        assert not anomalous_hours(wide, reference_days=28).to_numpy().any()


def test_hourly_table_has_values_and_flags_aligned(make_wide):
    wide = make_wide("2026-01-01", 60)
    wide.loc[45, "hour_03"] *= 0.2
    table = hourly_table(wide)
    assert list(table.columns) == ["timestamp", "value", "anomaly"]
    flagged = table[table["anomaly"]]
    assert list(flagged["timestamp"]) == [pd.Timestamp("2026-02-15 02:00", tz="America/Bogota")]
    assert flagged["value"].iloc[0] == wide.loc[45, "hour_03"]
