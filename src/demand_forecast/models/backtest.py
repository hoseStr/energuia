"""Walk-forward evaluation that only shows each forecast what was published when it was issued.

Days are timezone-aware timestamps at local midnight.
"""

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from demand_forecast import config


class Forecaster(Protocol):
    """A model that forecasts the 24 hours of a day from the history available at issue time."""

    name: str

    def fit(self, history: pd.Series) -> None: ...

    def predict(self, history: pd.Series, target_day: pd.Timestamp) -> pd.Series: ...


@dataclass(frozen=True)
class Fold:
    """One month of target days, forecast by a model fitted on data up to `train_end`."""

    train_end: pd.Timestamp
    target_days: pd.DatetimeIndex


def available_history(
    series: pd.Series, issue_day: pd.Timestamp, lag_days: int = config.DATA_LAG_DAYS
) -> pd.Series:
    """Return what a forecast issued on `issue_day` can see: data up to `issue_day - lag_days`.

    The stored history is complete, but in production the last `lag_days` days are not
    published yet. Every use of past data in an evaluation must go through this cut.
    """
    first_unpublished_hour = issue_day - pd.Timedelta(days=lag_days - 1)
    return series.iloc[: series.index.searchsorted(first_unpublished_hour, side="left")]


def evaluation_window(
    last_day: pd.Timestamp, months: int = config.BACKTEST_MONTHS
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the first and last day of the last `months` whole calendar months of data."""
    month_is_whole = (last_day + pd.Timedelta(days=1)).day == 1
    end = last_day if month_is_whole else last_day.replace(day=1) - pd.Timedelta(days=1)
    start = end + pd.Timedelta(days=1) - pd.DateOffset(months=months)
    return start, end


def make_folds(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    horizon_days: int = config.HORIZON_DAYS,
    lag_days: int = config.DATA_LAG_DAYS,
) -> list[Fold]:
    """Split [start, end] into one fold per calendar month.

    The model of a fold is refitted with everything available when the forecast for the first
    day of the fold is issued, so training data always ends before the fold starts.
    """
    days = pd.date_range(start, end, freq="D")
    folds = []
    for _, month in days.to_series().groupby([days.year, days.month]):
        target_days = pd.DatetimeIndex(month)
        first_issue_day = target_days[0] - pd.Timedelta(days=horizon_days)
        folds.append(Fold(first_issue_day - pd.Timedelta(days=lag_days), target_days))
    return folds


def run_backtest(
    series: pd.Series,
    forecaster: Forecaster,
    folds: list[Fold],
    *,
    horizon_days: int = config.HORIZON_DAYS,
    lag_days: int = config.DATA_LAG_DAYS,
) -> pd.DataFrame:
    """Forecast every target day of every fold; return `actual` and `predicted` by timestamp."""
    predictions = []
    for fold in folds:
        first_issue_day = fold.target_days[0] - pd.Timedelta(days=horizon_days)
        forecaster.fit(available_history(series, first_issue_day, lag_days))
        for target_day in fold.target_days:
            issue_day = target_day - pd.Timedelta(days=horizon_days)
            history = available_history(series, issue_day, lag_days)
            predictions.append(forecaster.predict(history, target_day))
    predicted = pd.concat(predictions).rename("predicted")
    missing = predicted.index.difference(series.index)
    if len(missing):
        raise ValueError(f"no actual values for {len(missing)} forecast hours, from {missing[0]}")
    return pd.DataFrame({"actual": series.loc[predicted.index], "predicted": predicted})
