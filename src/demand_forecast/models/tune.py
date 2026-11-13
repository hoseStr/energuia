"""Small hyperparameter search, scored on months the evaluation window never touches."""

from collections.abc import Callable, Mapping, Sequence
from itertools import product

import pandas as pd

from demand_forecast import config
from demand_forecast.models import metrics
from demand_forecast.models.backtest import Fold, run_backtest
from demand_forecast.models.train import LightGBMForecaster

FORECASTER_SETTINGS = ("train_years", "rounds")


def selection_window(
    evaluation_start: pd.Timestamp, months: int = config.BACKTEST_MONTHS
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the first and last day of the `months` months right before the evaluation window."""
    return evaluation_start - pd.DateOffset(months=months), evaluation_start - pd.Timedelta(days=1)


def candidates(grid: Mapping[str, Sequence[object]]) -> list[dict[str, object]]:
    """Return every combination of the values of `grid`."""
    return [dict(zip(grid, values, strict=True)) for values in product(*grid.values())]


def build_candidate(candidate: Mapping[str, object]) -> LightGBMForecaster:
    """Return the forecaster of a candidate: the configured one with its values replaced."""
    settings = {name: candidate[name] for name in FORECASTER_SETTINGS if name in candidate}
    params = {name: value for name, value in candidate.items() if name not in settings}
    return LightGBMForecaster({**config.LGBM_PARAMS, **params}, **settings)


def tune(
    demand: pd.Series,
    generation: pd.Series,
    folds: list[Fold],
    grid: Mapping[str, Sequence[object]] = config.TUNING_GRID,
    on_result: Callable[[dict[str, object]], None] | None = None,
) -> pd.DataFrame:
    """Backtest every candidate of `grid` on `folds`; return them sorted by MAPE, best first."""
    rows = []
    for candidate in candidates(grid):
        results = run_backtest(demand, build_candidate(candidate), folds, generation=generation)
        row = {**candidate, **metrics.overall(results).to_dict()}
        rows.append(row)
        if on_result is not None:
            on_result(row)
    return pd.DataFrame(rows).sort_values("mape", kind="stable", ignore_index=True)
