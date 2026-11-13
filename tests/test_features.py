import numpy as np
import pandas as pd
import pytest

from demand_forecast import config
from demand_forecast.features.build import (
    FEATURES,
    SCALE,
    TARGET,
    build_features,
    training_frame,
)
from demand_forecast.models.backtest import History, history_at

BOGOTA = config.TIMEZONE
DAY = pd.Timedelta(days=1)
LAGS = [(5, 3), (1, 1), (8, 2)]


def days(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="D", tz=BOGOTA)


def cut(targets: pd.DatetimeIndex, lag_days: int) -> pd.DatetimeIndex:
    """Last day published when the forecast of each target day is issued, one day ahead."""
    return targets - (1 + lag_days) * DAY


def build(demand, generation, targets, lag_days=5, generation_lag_days=3) -> pd.DataFrame:
    return build_features(
        demand,
        generation,
        targets,
        demand_cut=cut(targets, lag_days),
        generation_cut=cut(targets, generation_lag_days),
    )


def hours_of(day: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(day, periods=24, freq="h")


@pytest.fixture
def noisy(make_series) -> tuple[pd.Series, pd.Series]:
    """Demand and generation with noise, so that no two days are alike."""
    rng = np.random.default_rng(7)
    clean = make_series("2025-09-01", 200)
    return clean * rng.normal(1, 0.03, len(clean)), clean * rng.normal(1, 0.03, len(clean))


class TestNoLeak:
    # Target days around Christmas and New Year, so that holidays are in play.
    TARGETS = days("2025-12-18", 30)

    @pytest.mark.parametrize(("lag_days", "generation_lag_days"), LAGS)
    def test_rows_survive_destroying_everything_unpublished(
        self, noisy, lag_days, generation_lag_days
    ):
        demand, generation = noisy
        full = build(demand, generation, self.TARGETS, lag_days, generation_lag_days)

        for target in self.TARGETS:
            target = pd.DatetimeIndex([target])
            blind_demand, blind_generation = demand.copy(), generation.copy()
            blind_demand[blind_demand.index >= cut(target, lag_days)[0] + DAY] = np.nan
            blind_generation[
                blind_generation.index >= cut(target, generation_lag_days)[0] + DAY
            ] = np.nan

            rows = build(blind_demand, blind_generation, target, lag_days, generation_lag_days)

            assert not rows.isna().any().any()
            pd.testing.assert_frame_equal(rows, full.loc[rows.index])

    @pytest.mark.parametrize(("lag_days", "generation_lag_days"), LAGS)
    def test_rows_equal_those_built_from_the_history_cut_at_issue_time(
        self, noisy, lag_days, generation_lag_days
    ):
        demand, generation = noisy
        full = build(demand, generation, self.TARGETS, lag_days, generation_lag_days)

        for target in self.TARGETS:
            history = history_at(
                demand,
                generation,
                target - DAY,
                lag_days=lag_days,
                generation_lag_days=generation_lag_days,
            )
            rows = build(
                history.demand,
                history.generation,
                pd.DatetimeIndex([target]),
                lag_days,
                generation_lag_days,
            )
            pd.testing.assert_frame_equal(rows, full.loc[rows.index])

    def test_the_last_published_day_of_each_series_is_used(self, noisy):
        demand, generation = noisy
        target = days("2026-01-15", 1)
        honest = build(demand, generation, target)

        changed_demand = demand.copy()
        changed_demand[hours_of(cut(target, 5)[0])] *= 1.2
        changed_generation = generation.copy()
        changed_generation[hours_of(cut(target, 3)[0])] *= 1.2

        assert not build(changed_demand, generation, target).equals(honest)
        assert not build(demand, changed_generation, target).equals(honest)

    def test_the_first_unpublished_day_of_each_series_is_not_used(self, noisy):
        demand, generation = noisy
        target = days("2026-01-15", 1)
        honest = build(demand, generation, target)

        changed_demand = demand.copy()
        changed_demand[hours_of(cut(target, 5)[0] + DAY)] *= 1.2
        changed_generation = generation.copy()
        changed_generation[hours_of(cut(target, 3)[0] + DAY)] *= 1.2

        pd.testing.assert_frame_equal(build(changed_demand, changed_generation, target), honest)

    @pytest.mark.parametrize("series", ["demand_cut", "generation_cut"])
    @pytest.mark.parametrize("days_after_target", [0, 1])
    def test_a_cut_on_or_after_the_target_day_is_rejected(self, noisy, series, days_after_target):
        demand, generation = noisy
        target = days("2026-01-15", 1)
        cuts = {"demand_cut": cut(target, 5), "generation_cut": cut(target, 3)}
        cuts[series] = target + days_after_target * DAY
        with pytest.raises(ValueError, match="before its target day"):
            build_features(demand, generation, target, **cuts)

    def test_a_cut_that_the_history_does_not_reach_is_an_error(self, noisy):
        demand, generation = noisy
        target = days("2026-01-15", 1)
        with pytest.raises(ValueError, match="demand.*2026-01-09"):
            build(demand[:"2026-01-08"], generation, target)
        with pytest.raises(ValueError, match="generation.*2026-01-11"):
            build(demand, generation[:"2026-01-10"], target)

    def test_too_short_a_window_is_an_error(self, noisy):
        demand, generation = noisy
        with pytest.raises(ValueError, match="28 days of demand"):
            build(demand, generation, days("2025-09-30", 1))


class TestRows:
    def test_one_row_per_target_hour_in_order(self, noisy):
        demand, generation = noisy
        targets = days("2026-01-10", 3)
        rows = build(demand, generation, targets)
        assert list(rows.columns) == [*FEATURES, SCALE]
        assert list(rows.index) == list(pd.date_range(targets[0], periods=72, freq="h"))
        assert list(rows["hour"]) == list(range(24)) * 3

    def test_calendar_of_a_sunday_a_holiday_and_the_day_after(self, make_series):
        series = make_series("2026-08-01", 100)
        rows = build(series, series, days("2026-11-01", 3))
        by_day = rows[rows["hour"] == 0]
        assert list(by_day["weekday"]) == [6, 0, 1]
        assert list(by_day["day_type"]) == [2, 2, 0]
        assert list(by_day["is_holiday"]) == [0, 1, 0]
        assert list(by_day["holiday_eve"]) == [1, 0, 0]
        assert list(by_day["after_holiday"]) == [0, 0, 1]
        assert list(by_day["day_of_year"]) == [305, 306, 307]

    def test_a_purely_weekly_series_has_no_deviation_at_all(self, make_series):
        # No Colombian holiday between 2026-01-13 and 2026-03-22.
        series = make_series("2025-12-01", 120, growth=0.0)
        rows = build(series, series, days("2026-02-23", 14))

        ratios = [
            "same_weekday_last",
            "same_weekday_mean",
            "week_level",
            "demand_last_day_dev",
            "demand_last_hour_dev",
            "generation_last_day_dev",
            "generation_last_hour_dev",
            "generation_prev_day_dev",
            "generation_prev_hour_dev",
        ]
        for column in ratios:
            assert rows[column].to_numpy() == pytest.approx(1.0), column
        assert (rows["same_weekday_mismatch"] == 0).all()
        assert (rows["window_holidays"] == 0).all()
        assert set(rows.groupby("day_type")["same_type_days"].first().items()) == {
            (0, 20),
            (1, 4),
            (2, 4),
        }
        assert rows[SCALE].to_numpy() == pytest.approx(series.loc[rows.index].to_numpy())

    def test_a_holiday_among_the_same_weekdays_is_counted(self, make_series):
        series = make_series("2026-08-01", 100)
        # Monday 2026-11-09: its window holds the holiday Mondays 2026-10-12 and 2026-11-02.
        rows = build(series, series, days("2026-11-09", 1), lag_days=1, generation_lag_days=1)
        assert (rows["same_weekday_mismatch"] == 2).all()
        assert (rows["window_holidays"] == 2).all()

    def test_load_features_do_not_depend_on_the_level(self, noisy):
        demand, generation = noisy
        targets = days("2026-01-10", 5)
        small = build(demand, generation, targets)
        large = build(demand * 2, generation * 2, targets)
        pd.testing.assert_frame_equal(small[FEATURES], large[FEATURES])
        assert large[SCALE].to_numpy() == pytest.approx(2 * small[SCALE].to_numpy())


class TestTrainingFrame:
    def frame(self, noisy, issue_day: str = "2026-02-01", **kwargs) -> pd.DataFrame:
        demand, generation = noisy
        history = history_at(demand, generation, pd.Timestamp(issue_day, tz=BOGOTA))
        return training_frame(history, **kwargs)

    def test_targets_are_published_days_with_a_full_window_behind_them(self, noisy):
        frame = self.frame(noisy)
        demand, _ = noisy
        # 28 days of window end at the cut, which is 6 days before the first target.
        assert frame.index.min() == demand.index.min() + (27 + 6) * DAY
        assert frame.index.max() == pd.Timestamp("2026-01-27 23:00", tz=BOGOTA)
        assert list(frame.columns) == [*FEATURES, SCALE, TARGET]
        assert not frame.isna().any().any()

    def test_target_is_the_demand_relative_to_the_scale(self, noisy):
        frame = self.frame(noisy)
        demand, _ = noisy
        actual = demand.loc[frame.index].to_numpy()
        assert (frame[TARGET] * frame[SCALE]).to_numpy() == pytest.approx(actual)

    def test_every_row_is_built_as_if_it_had_been_forecast_live(self, noisy):
        frame = self.frame(noisy)
        demand, generation = noisy
        for target in days("2025-12-24", 3):
            rows = build(demand, generation, pd.DatetimeIndex([target]))
            pd.testing.assert_frame_equal(frame.loc[rows.index, [*FEATURES, SCALE]], rows)

    def test_first_target_drops_older_days(self, noisy):
        frame = self.frame(noisy, first_target=pd.Timestamp("2026-01-01", tz=BOGOTA))
        assert frame.index.min() == pd.Timestamp("2026-01-01", tz=BOGOTA)

    def test_generation_is_required(self, noisy):
        demand, _ = noisy
        with pytest.raises(ValueError, match="generation"):
            training_frame(History(demand=demand))
