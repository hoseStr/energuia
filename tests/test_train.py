import numpy as np
import pandas as pd
import pytest

from demand_forecast import config
from demand_forecast.models.backtest import Fold, History, history_at, make_folds, run_backtest
from demand_forecast.models.train import LightGBMForecaster

BOGOTA = config.TIMEZONE
DAY = pd.Timedelta(days=1)


def day(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=BOGOTA)


def small_model(**kwargs) -> LightGBMForecaster:
    return LightGBMForecaster(rounds=20, **kwargs)


@pytest.fixture
def noisy(make_series) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(3)
    clean = make_series("2025-06-01", 300)
    return clean * rng.normal(1, 0.03, len(clean)), clean * rng.normal(1, 0.03, len(clean))


class TestLightGBMForecaster:
    def test_forecasts_the_24_hours_of_the_target_day(self, noisy):
        demand, generation = noisy
        model = small_model()
        model.fit(history_at(demand, generation, day("2026-02-01")))

        forecast = model.predict(
            history_at(demand, generation, day("2026-02-09")), day("2026-02-10")
        )

        assert list(forecast.index) == list(pd.date_range(day("2026-02-10"), periods=24, freq="h"))
        actual = demand.loc[forecast.index]
        assert ((forecast - actual).abs() / actual).mean() < 0.1

    def test_forecasts_do_not_change_when_unpublished_data_changes(self, noisy):
        demand, generation = noisy
        target_day = day("2026-02-10")
        fold = [Fold(train_end=target_day - 6 * DAY, target_days=pd.DatetimeIndex([target_day]))]
        tampered_demand, tampered_generation = demand.copy(), generation.copy()
        tampered_demand[tampered_demand.index >= target_day - 5 * DAY] *= 3
        tampered_generation[tampered_generation.index >= target_day - 3 * DAY] *= 3

        honest = run_backtest(demand, small_model(), fold, generation=generation)
        blind = run_backtest(tampered_demand, small_model(), fold, generation=tampered_generation)

        assert list(honest["predicted"]) == list(blind["predicted"])

    def test_the_last_published_generation_day_matters(self, noisy):
        demand, generation = noisy
        target_day = day("2026-02-10")
        fold = [Fold(train_end=target_day - 6 * DAY, target_days=pd.DatetimeIndex([target_day]))]
        changed = generation.copy()
        changed[changed.index >= target_day - 4 * DAY] *= 1.1

        honest = run_backtest(demand, small_model(), fold, generation=generation)
        informed = run_backtest(demand, small_model(), fold, generation=changed)

        assert list(honest["predicted"]) != list(informed["predicted"])

    def test_two_fits_on_the_same_data_give_the_same_forecast(self, noisy):
        demand, generation = noisy
        folds = make_folds(day("2026-02-01"), day("2026-02-10"))
        first = run_backtest(demand, small_model(), folds, generation=generation)
        second = run_backtest(demand, small_model(), folds, generation=generation)
        assert list(first["predicted"]) == list(second["predicted"])

    def test_a_purely_weekly_series_is_forecast_without_error(self, make_series):
        # No Colombian holiday between 2026-01-13 and 2026-03-22.
        series = make_series("2025-06-01", 290, growth=0.0)
        folds = make_folds(day("2026-02-20"), day("2026-03-10"))
        results = run_backtest(series, small_model(), folds, generation=series)
        assert list(results["predicted"]) == pytest.approx(list(results["actual"]), rel=1e-3)

    def test_learns_only_from_the_last_train_years(self, make_series):
        series = make_series("2023-01-01", 3 * 365)
        history = history_at(series, series, day("2025-12-15"))
        model = small_model(train_years=1)
        model.fit(history)
        assert model.training_data.first_target_day == day("2024-12-11")
        assert model.training_data.last_target_day == day("2025-12-10")
        assert model.training_data.rows == 365 * 24

    def test_fit_without_generation_is_an_error(self, noisy):
        demand, _ = noisy
        with pytest.raises(ValueError, match="generation"):
            small_model().fit(History(demand=demand))

    def test_predict_without_generation_is_an_error(self, noisy):
        demand, generation = noisy
        model = small_model()
        model.fit(history_at(demand, generation, day("2026-02-01")))
        with pytest.raises(ValueError, match="2026-02-10"):
            model.predict(History(demand=demand[:"2026-02-04"]), day("2026-02-10"))

    def test_predict_before_fit_is_an_error(self, noisy):
        demand, generation = noisy
        with pytest.raises(ValueError, match="not been fitted"):
            small_model().predict(
                history_at(demand, generation, day("2026-02-09")), day("2026-02-10")
            )

    def test_too_short_a_history_is_an_error(self, make_series):
        series = make_series("2026-01-01", 30)
        with pytest.raises(ValueError, match="too short"):
            small_model().fit(History(demand=series, generation=series))
