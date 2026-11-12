"""Tell complete days from the partial ones XM publishes without any mark."""

from collections import deque
from statistics import median

import numpy as np
import pandas as pd

from demand_forecast import config


def complete_days(
    totals: pd.Series,
    *,
    min_ratio: float = config.COMPLETENESS_MIN_RATIO,
    reference_days: int = config.COMPLETENESS_REFERENCE_DAYS,
) -> pd.Series:
    """Flag each day as complete or not, given the daily totals sorted by date.

    A day is complete when its total reaches `min_ratio` of the median of the previous
    `reference_days` complete days. A day with a missing hour (NaN total) is never complete.
    The first `reference_days` days have no reference and are taken as complete: the start of
    XM's history was checked by hand (docs/data_notes.md section 4).
    """
    reference: deque[float] = deque(maxlen=reference_days)
    flags = []
    for total in totals.to_numpy():
        if np.isnan(total):
            complete = False
        elif len(reference) < reference_days:
            complete = True
        else:
            complete = total >= min_ratio * median(reference)
        if complete:
            reference.append(total)
        flags.append(complete)
    return pd.Series(flags, index=totals.index, dtype=bool)


def confirmed_by_generation(
    demand: pd.Series,
    generation: pd.Series,
    generation_complete: pd.Series,
    *,
    min_ratio: float = config.DEMAND_VS_GENERATION_MIN_RATIO,
) -> pd.Series:
    """Flag the days whose demand reaches `min_ratio` of that day's complete generation.

    Demand and generation match almost exactly, and generation is published two days earlier.
    A day without complete generation cannot be confirmed.
    """
    generation = generation.where(generation_complete).reindex(demand.index)
    return (demand >= min_ratio * generation).astype(bool)


def last_gapless_day(complete: pd.Series) -> pd.Timestamp | None:
    """Return the last day of the run of consecutive complete days that starts at the first day."""
    if complete.empty:
        return None
    days = pd.date_range(complete.index[0], complete.index[-1], freq="D")
    flags = complete.reindex(days, fill_value=False).to_numpy()
    if flags.all():
        return days[-1]
    first_gap = int(np.argmin(flags))
    return days[first_gap - 1] if first_gap > 0 else None
