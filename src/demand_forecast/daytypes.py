"""Day types used to break the forecast error down."""

import holidays
import pandas as pd

from demand_forecast import config

WORKDAY = "workday"
SATURDAY = "saturday"
SUNDAY_OR_HOLIDAY = "sunday_or_holiday"
DAY_TYPES = [WORKDAY, SATURDAY, SUNDAY_OR_HOLIDAY]


def day_types(days: pd.DatetimeIndex) -> pd.Series:
    """Return the day type of each day, indexed by day. A holiday wins over the weekday."""
    calendar = holidays.country_holidays(config.HOLIDAY_COUNTRY, years=sorted(set(days.year)))
    types = pd.Series(WORKDAY, index=days)
    types[days.dayofweek == 5] = SATURDAY
    types[(days.dayofweek == 6) | [day in calendar for day in days.date]] = SUNDAY_OR_HOLIDAY
    return types
