"""Reshape the wide table XM returns (one row per day) into an hourly series."""

import numpy as np
import pandas as pd

from demand_forecast import config
from demand_forecast.data.schema import HOUR_COLUMNS, TIMESTAMP_DTYPE


def daily_totals(wide: pd.DataFrame) -> pd.Series:
    """Return the kWh of each day, indexed by date. NaN when any hour is missing."""
    return wide.set_index("date")[HOUR_COLUMNS].sum(axis=1, min_count=len(HOUR_COLUMNS))


def wide_to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Return one row per hour with columns `timestamp` and `value`, sorted by time.

    `hour_NN` covers [NN-1:00, NN:00) in local time, so its start-of-hour timestamp is
    `date + (NN-1) hours` in America/Bogota (docs/data_notes.md section 2).
    """
    wide = wide.sort_values("date")
    days = pd.DatetimeIndex(wide["date"]).repeat(len(HOUR_COLUMNS))
    offsets = pd.to_timedelta(np.tile(np.arange(len(HOUR_COLUMNS)), len(wide)), unit="h")
    timestamps = (days + offsets).tz_localize(config.TIMEZONE).astype(TIMESTAMP_DTYPE)
    return pd.DataFrame({"timestamp": timestamps, "value": wide[HOUR_COLUMNS].to_numpy().ravel()})


def anomalous_hours(
    wide: pd.DataFrame,
    *,
    reference_days: int = config.ANOMALY_REFERENCE_DAYS,
    max_deviation: float = config.ANOMALY_MAX_DEVIATION,
) -> pd.DataFrame:
    """Flag the hours that deviate more than `max_deviation` from their recent level.

    The level is the median of the same hour over the previous `reference_days` days. It only
    looks backwards, so the flag of a day never depends on data published after it.
    """
    hours = wide.sort_values("date").set_index("date")[HOUR_COLUMNS]
    reference = hours.rolling(reference_days).median().shift(1)
    return (hours / reference - 1).abs() > max_deviation


def hourly_table(wide: pd.DataFrame) -> pd.DataFrame:
    """Return the hourly table (`timestamp`, `value`, `anomaly`) of a wide table."""
    table = wide_to_long(wide)
    table["anomaly"] = anomalous_hours(wide).to_numpy().ravel()
    return table
