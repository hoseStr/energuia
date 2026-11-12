import numpy as np
import pandas as pd

from demand_forecast.data.completeness import (
    complete_days,
    confirmed_by_generation,
    last_gapless_day,
)
from demand_forecast.data.transform import daily_totals

PARTIAL = ["2026-11-06", "2026-11-07"]


def flat_totals(days: int, level: float = 240e6) -> pd.Series:
    return pd.Series(level, index=pd.date_range("2026-01-01", periods=days, freq="D"))


class TestCompleteDays:
    def test_real_partial_days_are_rejected(self, demand_wide):
        complete = complete_days(daily_totals(demand_wide))
        assert not complete.loc[PARTIAL].any()
        assert complete.drop(pd.to_datetime(PARTIAL)).all()

    def test_real_sunday_and_holiday_are_complete(self, demand_wide):
        totals = daily_totals(demand_wide)
        complete = complete_days(totals)
        # Both sit below 90 % of the 28-day median, which is why the threshold is not 0.9.
        assert complete.loc["2026-11-01"]
        assert complete.loc["2026-11-02"]
        assert not complete_days(totals, min_ratio=0.9).loc["2026-11-01"]

    def test_generation_fixture_is_complete_through_the_last_day(self, generation_wide):
        assert complete_days(daily_totals(generation_wide)).all()

    def test_a_day_with_a_missing_hour_is_incomplete(self, demand_wide):
        demand_wide.loc[35, "hour_10"] = np.nan
        complete = complete_days(daily_totals(demand_wide))
        assert not complete.iloc[35]

    def test_partial_days_never_enter_the_reference(self):
        totals = flat_totals(100)
        totals.iloc[40:] *= 0.3
        complete = complete_days(totals)
        assert complete.iloc[:40].all()
        assert not complete.iloc[40:].any()

    def test_the_first_days_have_no_reference_and_are_accepted(self):
        totals = flat_totals(40)
        totals.iloc[5] *= 0.3
        totals.iloc[35] *= 0.3
        complete = complete_days(totals, reference_days=28)
        assert complete.iloc[5]
        assert not complete.iloc[35]

    def test_threshold_is_inclusive(self):
        totals = flat_totals(30)
        totals.iloc[29] = 0.6 * 240e6
        assert complete_days(totals, min_ratio=0.6).iloc[29]
        totals.iloc[29] -= 1
        assert not complete_days(totals, min_ratio=0.6).iloc[29]


class TestConfirmedByGeneration:
    def test_real_partial_days_are_not_confirmed(self, demand_wide, generation_wide):
        generation = daily_totals(generation_wide)
        confirmed = confirmed_by_generation(
            daily_totals(demand_wide), generation, complete_days(generation)
        )
        assert not confirmed.loc[PARTIAL].any()
        assert confirmed.drop(pd.to_datetime(PARTIAL)).all()

    def test_catches_a_half_filled_day_that_the_total_alone_lets_through(
        self, demand_wide, generation_wide
    ):
        # Regulated market complete, non-regulated missing: about 70 % of a normal day.
        demand = daily_totals(demand_wide)
        demand.loc["2026-11-05"] *= 0.70
        generation = daily_totals(generation_wide)

        assert complete_days(demand).loc["2026-11-05"]
        confirmed = confirmed_by_generation(demand, generation, complete_days(generation))
        assert not confirmed.loc["2026-11-05"]

    def test_a_day_without_complete_generation_is_not_confirmed(self):
        demand = flat_totals(5)
        generation = flat_totals(4)
        generation_complete = pd.Series([True, True, False, True], index=generation.index)
        confirmed = confirmed_by_generation(demand, generation, generation_complete)
        assert list(confirmed) == [True, True, False, True, False]


class TestLastGaplessDay:
    DAYS = pd.date_range("2026-11-01", periods=6, freq="D")

    def flags(self, values: list[bool]) -> pd.Series:
        return pd.Series(values, index=self.DAYS[: len(values)])

    def test_all_complete(self):
        assert last_gapless_day(self.flags([True] * 6)) == pd.Timestamp("2026-11-06")

    def test_trailing_partial_days(self):
        flags = self.flags([True, True, True, True, False, False])
        assert last_gapless_day(flags) == pd.Timestamp("2026-11-04")

    def test_stops_before_a_hole_even_if_later_days_are_complete(self):
        flags = self.flags([True, True, False, True, True, True])
        assert last_gapless_day(flags) == pd.Timestamp("2026-11-02")

    def test_a_date_missing_from_the_index_is_a_hole(self):
        flags = self.flags([True] * 6).drop(pd.Timestamp("2026-11-04"))
        assert last_gapless_day(flags) == pd.Timestamp("2026-11-03")

    def test_nothing_usable(self):
        assert last_gapless_day(self.flags([False, True, True])) is None
        assert last_gapless_day(pd.Series(dtype=bool)) is None
