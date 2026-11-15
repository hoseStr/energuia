"""Decide whether retraining pays: retrained models against the current one, out of sample.

The current model is frozen: it never saw the days after its training. Over those days it is
compared with models refitted every week on what was published at the time, which is what a
retraining routine would have produced. Comparing a model fitted today on those same days
would be unfair, because it has already seen them.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from demand_forecast import config
from demand_forecast.models import metrics
from demand_forecast.models.backtest import Fold, Forecaster, History, run_backtest
from demand_forecast.models.train import LightGBMForecaster


@dataclass(frozen=True)
class Comparison:
    """Error of the current model and of weekly retrained ones over the same days."""

    start: pd.Timestamp | None
    end: pd.Timestamp | None
    days: int
    champion_mape: float | None
    challenger_mape: float | None


class _Frozen:
    """A fitted model that the backtest must not refit."""

    name = "champion"

    def __init__(self, model: Forecaster) -> None:
        self._model = model

    def fit(self, history: History) -> None:
        """Keep the model as it is."""

    def predict(self, history: History, target_day: pd.Timestamp) -> pd.Series:
        return self._model.predict(history, target_day)


def compare(
    demand: pd.Series,
    generation: pd.Series,
    champion: Forecaster,
    trained_up_to: pd.Timestamp,
    *,
    build: Callable[[], Forecaster] = LightGBMForecaster,
    refit_days: int = config.RETRAIN_REFIT_DAYS,
    horizon_days: int = config.HORIZON_DAYS,
    lag_days: int = config.DATA_LAG_DAYS,
) -> Comparison:
    """Score `champion` and weekly refitted models on every published day after `trained_up_to`.

    `trained_up_to` is the last day the champion learned from. Each forecast, of either side,
    only sees what was published when it would have been issued.
    """
    days = pd.date_range(
        trained_up_to + pd.Timedelta(days=1), demand.index[-1].normalize(), freq="D"
    )
    if days.empty:
        return Comparison(None, None, 0, None, None)

    def fold(target_days: pd.DatetimeIndex) -> Fold:
        first_issue_day = target_days[0] - pd.Timedelta(days=horizon_days)
        return Fold(first_issue_day - pd.Timedelta(days=lag_days), target_days)

    weekly = [fold(days[start : start + refit_days]) for start in range(0, len(days), refit_days)]
    frozen = run_backtest(demand, _Frozen(champion), [fold(days)], generation=generation)
    retrained = run_backtest(demand, build(), weekly, generation=generation)
    return Comparison(
        start=days[0],
        end=days[-1],
        days=len(days),
        champion_mape=float(metrics.overall(frozen)["mape"]),
        challenger_mape=float(metrics.overall(retrained)["mape"]),
    )


def decide(comparison: Comparison, *, min_days: int = config.RETRAIN_MIN_DAYS) -> tuple[bool, str]:
    """Return whether to promote a retrained model, and why."""
    if comparison.days < min_days:
        return False, (
            f"only {comparison.days} of the {min_days} days needed have been published since "
            "the current model was trained"
        )
    if comparison.challenger_mape < comparison.champion_mape:
        return True, "models retrained every week beat the current model over those days"
    return False, "the current model was at least as good as the ones retrained every week"
