import pandas as pd
import pytest

from demand_forecast import config
from demand_forecast.models.backtest import (
    Fold,
    History,
    available_history,
    evaluation_window,
    make_folds,
    run_backtest,
)
from demand_forecast.models.baselines import BASELINES, MovingAverage, SeasonalNaive

BOGOTA = "America/Bogota"
DAY = pd.Timedelta(days=1)


def day(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=BOGOTA)


class Spy:
    """Forecaster that records what it was shown and predicts a constant."""

    name = "spy"

    def __init__(self) -> None:
        self.fits: list[pd.Timestamp] = []
        self.predictions: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        self.generation_fits: list[pd.Timestamp] = []
        self.generation_predictions: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def fit(self, history: History) -> None:
        self.fits.append(history.demand.index.max())
        if history.generation is not None:
            self.generation_fits.append(history.generation.index.max())

    def predict(self, history: History, target_day: pd.Timestamp) -> pd.Series:
        self.predictions.append((target_day, history.demand.index.max()))
        if history.generation is not None:
            self.generation_predictions.append((target_day, history.generation.index.max()))
        return pd.Series(1.0, index=pd.date_range(target_day, periods=24, freq="h"))


class TestAvailableHistory:
    def test_ends_at_the_last_hour_of_the_last_published_day(self, make_series):
        series = make_series("2026-01-01", 60)
        history = available_history(series, day("2026-02-20"), lag_days=5)
        assert history.index.max() == pd.Timestamp("2026-02-15 23:00", tz=BOGOTA)
        assert history.index.min() == series.index.min()

    def test_default_lag_is_the_configured_one(self, make_series):
        series = make_series("2026-01-01", 60)
        history = available_history(series, day("2026-02-20"))
        assert history.index.max().normalize() == day("2026-02-20") - config.DATA_LAG_DAYS * DAY

    def test_issue_day_before_any_data_gives_an_empty_history(self, make_series):
        series = make_series("2026-01-01", 60)
        assert available_history(series, day("2026-01-03"), lag_days=5).empty


class TestEvaluationWindow:
    def test_last_twelve_whole_months_when_the_data_ends_mid_month(self):
        start, end = evaluation_window(day("2026-11-06"), months=12)
        assert (start, end) == (day("2025-11-01"), day("2026-10-31"))

    def test_a_month_that_just_ended_counts(self):
        start, end = evaluation_window(day("2026-10-31"), months=12)
        assert (start, end) == (day("2025-11-01"), day("2026-10-31"))

    def test_window_does_not_move_within_a_month(self):
        assert evaluation_window(day("2026-11-01")) == evaluation_window(day("2026-11-29"))

    def test_leap_february(self):
        start, end = evaluation_window(day("2024-03-10"), months=1)
        assert (start, end) == (day("2024-02-01"), day("2024-02-29"))


class TestMakeFolds:
    START, END = day("2025-11-01"), day("2026-10-31")

    def test_one_fold_per_calendar_month(self):
        folds = make_folds(self.START, self.END)
        assert len(folds) == 12
        for fold in folds:
            assert fold.target_days[0].day == 1
            assert len({(d.year, d.month) for d in fold.target_days}) == 1

    def test_every_target_day_is_forecast_exactly_once(self):
        folds = make_folds(self.START, self.END)
        targets = [d for fold in folds for d in fold.target_days]
        assert targets == list(pd.date_range(self.START, self.END, freq="D"))

    def test_training_data_ends_before_the_fold_with_the_publication_gap(self):
        for fold in make_folds(self.START, self.END, horizon_days=1, lag_days=5):
            assert fold.train_end == fold.target_days[0] - 6 * DAY

    def test_partial_months_at_the_edges_are_their_own_folds(self):
        folds = make_folds(day("2026-01-20"), day("2026-03-05"))
        assert [len(fold.target_days) for fold in folds] == [12, 28, 5]


class TestRunBacktestDoesNotLeak:
    def run(self, make_series, lag_days: int = 5) -> tuple[Spy, list[Fold], pd.DataFrame]:
        series = make_series("2025-10-01", 150)
        folds = make_folds(day("2025-12-01"), day("2026-01-31"), lag_days=lag_days)
        spy = Spy()
        results = run_backtest(series, spy, folds, lag_days=lag_days)
        return spy, folds, results

    @pytest.mark.parametrize("lag_days", [1, 5, 8])
    def test_predict_never_sees_data_published_after_the_issue_day(self, make_series, lag_days):
        spy, _, _ = self.run(make_series, lag_days)
        assert len(spy.predictions) == 62
        for target_day, last_seen in spy.predictions:
            issue_day = target_day - DAY
            assert last_seen == issue_day - lag_days * DAY + pd.Timedelta(hours=23)

    def test_fit_only_sees_data_up_to_the_train_end_of_its_fold(self, make_series):
        spy, folds, _ = self.run(make_series)
        assert spy.fits == [fold.train_end + pd.Timedelta(hours=23) for fold in folds]
        for fold, last_fitted in zip(folds, spy.fits, strict=True):
            assert last_fitted < fold.target_days[0]

    @pytest.mark.parametrize("generation_lag_days", [1, 3, 5])
    def test_generation_is_cut_at_its_own_lag(self, make_series, generation_lag_days):
        series = make_series("2025-10-01", 150)
        folds = make_folds(day("2025-12-01"), day("2026-01-31"))
        spy = Spy()

        run_backtest(
            series, spy, folds, generation=series * 1.01, generation_lag_days=generation_lag_days
        )

        last_hour = pd.Timedelta(hours=23) - generation_lag_days * DAY
        assert len(spy.generation_predictions) == 62
        for target_day, last_seen in spy.generation_predictions:
            assert last_seen == target_day - DAY + last_hour
        first_issue_days = [fold.target_days[0] - DAY for fold in folds]
        assert spy.generation_fits == [issue_day + last_hour for issue_day in first_issue_days]

    def test_without_generation_the_forecaster_gets_none(self, make_series):
        spy, _, _ = self.run(make_series)
        assert spy.generation_fits == []
        assert spy.generation_predictions == []

    @pytest.mark.parametrize("model", BASELINES.values(), ids=BASELINES.keys())
    def test_forecasts_do_not_change_when_unpublished_data_changes(self, make_series, model):
        series = make_series("2025-10-01", 150)
        target_day = day("2026-01-15")
        fold = [Fold(train_end=target_day - 6 * DAY, target_days=pd.DatetimeIndex([target_day]))]
        tampered = series.copy()
        tampered[tampered.index >= target_day - 5 * DAY] *= 3

        honest = run_backtest(series, model(), fold)
        blind = run_backtest(tampered, model(), fold)

        assert list(honest["predicted"]) == list(blind["predicted"])
        assert list(blind["actual"]) == list(honest["actual"] * 3)


class TestRunBacktestResults:
    def test_one_row_per_target_hour_with_the_real_value(self, make_series):
        series = make_series("2025-10-01", 150)
        folds = make_folds(day("2025-12-01"), day("2026-01-31"))

        results = run_backtest(series, Spy(), folds)

        assert list(results.columns) == ["actual", "predicted"]
        assert len(results) == 62 * 24
        assert results.index.min() == day("2025-12-01")
        assert results.index.is_unique
        assert results["actual"].equals(series.loc[results.index])
        assert (results["predicted"] == 1.0).all()

    def test_target_days_without_actual_data_are_an_error(self, make_series):
        series = make_series("2025-10-01", 60)
        folds = make_folds(day("2025-11-25"), day("2025-12-05"))
        with pytest.raises(ValueError, match="no actual values"):
            run_backtest(series, Spy(), folds)


class TestBaselines:
    def predict(self, model, series: pd.Series, target: str, lag_days: int = 5) -> pd.Series:
        history = History(demand=available_history(series, day(target) - DAY, lag_days))
        return model.predict(history, day(target))

    def values(self, series: pd.Series, target: str) -> list[float]:
        return list(series.loc[day(target) : day(target) + pd.Timedelta(hours=23)])

    def test_seasonal_naive_repeats_the_same_weekday_of_last_week(self, make_series):
        series = make_series("2026-01-01", 90)
        forecast = self.predict(SeasonalNaive(), series, "2026-03-10")
        assert list(forecast) == self.values(series, "2026-03-03")
        assert list(forecast.index) == list(pd.date_range(day("2026-03-10"), periods=24, freq="h"))

    def test_seasonal_naive_goes_further_back_when_last_week_is_not_published(self, make_series):
        series = make_series("2026-01-01", 90)
        forecast = self.predict(SeasonalNaive(), series, "2026-03-10", lag_days=7)
        assert list(forecast) == self.values(series, "2026-02-24")

    def test_moving_average_uses_the_last_four_published_weeks(self, make_series):
        series = make_series("2026-01-01", 90)
        forecast = self.predict(MovingAverage(weeks=4), series, "2026-03-10")
        weeks = ["2026-03-03", "2026-02-24", "2026-02-17", "2026-02-10"]
        expected = pd.DataFrame([self.values(series, week) for week in weeks]).mean()
        assert list(forecast) == pytest.approx(list(expected))

    def test_moving_average_skips_weeks_that_are_not_published(self, make_series):
        series = make_series("2026-01-01", 90)
        forecast = self.predict(MovingAverage(weeks=2), series, "2026-03-10", lag_days=7)
        weeks = ["2026-02-24", "2026-02-17"]
        expected = pd.DataFrame([self.values(series, week) for week in weeks]).mean()
        assert list(forecast) == pytest.approx(list(expected))

    def test_a_purely_weekly_series_is_forecast_without_error(self, make_series):
        series = make_series("2025-10-01", 150, growth=0.0)
        folds = make_folds(day("2025-12-01"), day("2026-01-31"))
        for model in (SeasonalNaive(), MovingAverage()):
            results = run_backtest(series, model, folds)
            assert list(results["predicted"]) == pytest.approx(list(results["actual"]))

    def test_no_history_at_all_is_an_error(self, make_series):
        series = make_series("2026-01-01", 20)
        with pytest.raises(ValueError, match="no published history to forecast 2026-01-03"):
            self.predict(SeasonalNaive(), series, "2026-01-03")

    def test_too_short_a_history_is_an_error(self, make_series):
        series = make_series("2026-01-01", 20)
        with pytest.raises(ValueError, match="24 hours of 2025-12-30"):
            self.predict(MovingAverage(weeks=4), series, "2026-01-27")
