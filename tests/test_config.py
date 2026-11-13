"""The configured publication lags must not promise data XM had not published yet."""

import pandas as pd

from demand_forecast import config
from demand_forecast.data.completeness import (
    complete_days,
    confirmed_by_generation,
    last_gapless_day,
)
from demand_forecast.data.transform import daily_totals
from demand_forecast.models.backtest import available_history

# Day the fixtures were downloaded from XM (tests/conftest.py).
DOWNLOADED_ON = pd.Timestamp("2026-11-10")


def measured_lag(complete: pd.Series) -> int:
    return (DOWNLOADED_ON - last_gapless_day(complete)).days


class TestPublicationLags:
    def test_demand_lag_covers_the_measured_one(self, demand_wide, generation_wide):
        demand, generation = daily_totals(demand_wide), daily_totals(generation_wide)
        complete = complete_days(demand) & confirmed_by_generation(
            demand, generation, complete_days(generation)
        )
        assert measured_lag(complete) == 5
        assert measured_lag(complete) <= config.DATA_LAG_DAYS

    def test_generation_lag_covers_the_measured_one(self, generation_wide):
        complete = complete_days(daily_totals(generation_wide))
        assert measured_lag(complete) == 3
        assert measured_lag(complete) <= config.GENERATION_LAG_DAYS

    def test_generation_is_never_assumed_to_lag_more_than_demand(self):
        assert config.GENERATION_LAG_DAYS <= config.DATA_LAG_DAYS

    def test_generation_cut_ends_at_its_own_last_published_day(self, make_series):
        generation = make_series("2026-01-01", 60)
        issue_day = pd.Timestamp("2026-02-20", tz=config.TIMEZONE)
        history = available_history(generation, issue_day, config.GENERATION_LAG_DAYS)
        last_published_day = issue_day - pd.Timedelta(days=config.GENERATION_LAG_DAYS)
        assert history.index.max() == last_published_day + pd.Timedelta(hours=23)
