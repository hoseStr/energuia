import json

import numpy as np
import pandas as pd
import pytest

from demand_forecast import config
from demand_forecast.models.backtest import History
from demand_forecast.models.promote import compare, decide
from demand_forecast.models.registry import read_registry, record_decision
from demand_forecast.models.train import LightGBMForecaster

BOGOTA = config.TIMEZONE
ONE_DAY = pd.Timedelta(days=1)


def day(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=BOGOTA)


@pytest.fixture(scope="module")
def series() -> tuple[pd.Series, pd.Series]:
    """Noisy synthetic demand and generation from 2026-01-01 to 2026-11-20."""
    rng = np.random.default_rng(21)
    index = pd.date_range(day("2026-01-01"), day("2026-11-20 23:00"), freq="h")
    shape = 1 + 0.2 * np.sin(index.hour / 24 * 2 * np.pi)
    clean = pd.Series(1e7 * shape * np.where(index.dayofweek == 6, 0.85, 1.0), index=index)
    return clean * rng.normal(1, 0.03, len(clean)), clean * rng.normal(1, 0.03, len(clean))


def quick_model() -> LightGBMForecaster:
    return LightGBMForecaster(rounds=10, train_years=1)


class Biased:
    """A model whose forecast is the demand it should not know, off by a fixed share."""

    def __init__(self, demand: pd.Series, bias: float) -> None:
        self._demand, self._bias = demand, bias
        self.seen: list[pd.Timestamp] = []

    def predict(self, history: History, target_day: pd.Timestamp) -> pd.Series:
        self.seen.append(history.demand.index[-1].normalize())
        hours = pd.date_range(target_day, periods=24, freq="h")
        return self._demand.loc[hours] * (1 + self._bias)


def test_the_window_is_every_published_day_after_the_training_of_the_current_model(series):
    demand, generation = series
    champion = Biased(demand, 0.10)

    comparison = compare(demand, generation, champion, day("2026-10-20"), build=quick_model)

    assert comparison.start == day("2026-10-21")
    assert comparison.end == day("2026-11-20")
    assert comparison.days == 31
    assert comparison.champion_mape == pytest.approx(10.0)
    assert 0 < comparison.challenger_mape < 10


def test_the_current_model_only_sees_what_was_published_for_each_day(series):
    demand, generation = series
    champion = Biased(demand, 0.10)

    compare(demand, generation, champion, day("2026-11-10"), build=quick_model)

    lag = config.DATA_LAG_DAYS + config.HORIZON_DAYS
    assert champion.seen == [
        target - lag * ONE_DAY
        for target in pd.date_range(day("2026-11-11"), day("2026-11-20"), freq="D")
    ]


def test_retrained_models_are_refitted_every_week_with_what_was_published(series):
    demand, generation = series
    fitted_up_to = []

    class Spy(LightGBMForecaster):
        def __init__(self) -> None:
            super().__init__(rounds=5, train_years=1)

        def fit(self, history: History) -> None:
            fitted_up_to.append(history.demand.index[-1].normalize())
            super().fit(history)

    compare(demand, generation, Biased(demand, 0.1), day("2026-10-30"), build=Spy)

    lag = config.DATA_LAG_DAYS + config.HORIZON_DAYS
    assert fitted_up_to == [
        day("2026-10-31") - lag * ONE_DAY,
        day("2026-11-07") - lag * ONE_DAY,
        day("2026-11-14") - lag * ONE_DAY,
    ]


def test_without_days_after_the_training_there_is_nothing_to_compare(series):
    demand, generation = series

    comparison = compare(
        demand, generation, Biased(demand, 0.1), day("2026-11-20"), build=quick_model
    )

    assert comparison.days == 0
    assert comparison.start is None
    assert comparison.champion_mape is None
    assert comparison.challenger_mape is None


def test_a_better_retrained_model_is_promoted_once_there_are_enough_days(series):
    demand, generation = series
    comparison = compare(
        demand, generation, Biased(demand, 0.10), day("2026-10-20"), build=quick_model
    )

    promote, reason = decide(comparison, min_days=28)

    assert promote
    assert "beat the current model" in reason


def test_the_current_model_stays_when_it_is_at_least_as_good(series):
    demand, generation = series
    comparison = compare(
        demand, generation, Biased(demand, 0.0001), day("2026-10-20"), build=quick_model
    )

    promote, reason = decide(comparison, min_days=28)

    assert not promote
    assert "at least as good" in reason


def test_nothing_is_promoted_on_too_few_days(series):
    demand, generation = series
    comparison = compare(
        demand, generation, Biased(demand, 0.10), day("2026-11-10"), build=quick_model
    )

    promote, reason = decide(comparison, min_days=28)

    assert comparison.days == 10
    assert not promote
    assert "10 of the 28 days" in reason


def test_every_decision_is_kept_in_the_registry(tmp_path):
    (tmp_path / "registry.json").write_text(
        json.dumps({"current": "20261113T163005Z", "models": [{"version": "20261113T163005Z"}]}),
        encoding="utf-8",
    )

    record_decision({"decided_at": "2026-11-22T12:17:00+00:00", "promoted": False}, tmp_path)
    record_decision({"decided_at": "2026-11-29T12:17:00+00:00", "promoted": True}, tmp_path)

    registry = read_registry(tmp_path)
    assert registry["current"] == "20261113T163005Z"
    assert registry["models"] == [{"version": "20261113T163005Z"}]
    assert [one["promoted"] for one in registry["decisions"]] == [False, True]
