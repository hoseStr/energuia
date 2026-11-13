"""Day types used to break the forecast error down and to build calendar features."""

from datetime import date
from functools import cache

import holidays
import numpy as np
import pandas as pd

from demand_forecast import config

WORKDAY = "workday"
SATURDAY = "saturday"
SUNDAY_OR_HOLIDAY = "sunday_or_holiday"
DAY_TYPES = [WORKDAY, SATURDAY, SUNDAY_OR_HOLIDAY]


@cache
def _holidays_of(year: int) -> frozenset[date]:
    return frozenset(holidays.country_holidays(config.HOLIDAY_COUNTRY, years=year))


def is_holiday(days: pd.DatetimeIndex) -> np.ndarray:
    """Return, for each day, whether it is a Colombian public holiday."""
    found = (day in _holidays_of(day.year) for day in days.date)
    return np.fromiter(found, dtype=bool, count=len(days))


def day_type_codes(days: pd.DatetimeIndex) -> np.ndarray:
    """Return the position in `DAY_TYPES` of the type of each day. A holiday wins."""
    codes = np.full(len(days), DAY_TYPES.index(WORKDAY))
    codes[days.dayofweek == 5] = DAY_TYPES.index(SATURDAY)
    codes[(days.dayofweek == 6) | is_holiday(days)] = DAY_TYPES.index(SUNDAY_OR_HOLIDAY)
    return codes


def day_types(days: pd.DatetimeIndex) -> pd.Series:
    """Return the day type of each day, indexed by day. A holiday wins over the weekday."""
    return pd.Series(np.array(DAY_TYPES)[day_type_codes(days)], index=days)
