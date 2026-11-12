"""Baselines every model has to beat (PLAN.md section 4)."""

import math

import pandas as pd

from demand_forecast import config

HOURS_PER_DAY = 24


def _same_weekday_days(
    history: pd.Series, target_day: pd.Timestamp, count: int
) -> list[pd.Timestamp]:
    """Return the `count` most recent days in `history` with the weekday of `target_day`."""
    if history.empty:
        raise ValueError(f"no published history to forecast {target_day.date()}")
    last_day = history.index.max().normalize()
    first_week = max(1, math.ceil((target_day - last_day).days / 7))
    return [target_day - pd.Timedelta(weeks=first_week + i) for i in range(count)]


def _day_values(history: pd.Series, day: pd.Timestamp) -> list[float]:
    values = history.loc[day : day + pd.Timedelta(hours=HOURS_PER_DAY - 1)]
    if len(values) != HOURS_PER_DAY:
        raise ValueError(f"history does not have the 24 hours of {day.date()}")
    return values.to_list()


def _target_hours(target_day: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(target_day, periods=HOURS_PER_DAY, freq="h")


class SeasonalNaive:
    """Repeat the most recent available day with the same weekday as the target."""

    name = "seasonal-naive"

    def fit(self, history: pd.Series) -> None:
        """Nothing to learn."""

    def predict(self, history: pd.Series, target_day: pd.Timestamp) -> pd.Series:
        """Return the 24 hourly values forecast for `target_day`."""
        (day,) = _same_weekday_days(history, target_day, 1)
        return pd.Series(_day_values(history, day), index=_target_hours(target_day))


class MovingAverage:
    """Average, hour by hour, the last available weeks with the same weekday as the target."""

    name = "moving-average"

    def __init__(self, weeks: int = config.MOVING_AVERAGE_WEEKS) -> None:
        self._weeks = weeks

    def fit(self, history: pd.Series) -> None:
        """Nothing to learn."""

    def predict(self, history: pd.Series, target_day: pd.Timestamp) -> pd.Series:
        """Return the 24 hourly values forecast for `target_day`."""
        days = _same_weekday_days(history, target_day, self._weeks)
        values = pd.DataFrame([_day_values(history, day) for day in days])
        return pd.Series(values.mean().to_numpy(), index=_target_hours(target_day))


BASELINES = {model.name: model for model in (SeasonalNaive, MovingAverage)}
