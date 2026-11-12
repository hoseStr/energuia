"""Forecast error metrics, overall and broken down by hour and by day type.

Every function takes a table indexed by timestamp with columns `actual` and `predicted`.
"""

import pandas as pd

from demand_forecast.daytypes import DAY_TYPES, day_types


def _errors(results: pd.DataFrame) -> pd.DataFrame:
    absolute = (results["predicted"] - results["actual"]).abs()
    return pd.DataFrame({"mae": absolute, "mape": absolute / results["actual"] * 100})


def overall(results: pd.DataFrame) -> pd.Series:
    """Return the MAE (in the unit of the series) and the MAPE (in percent)."""
    return _errors(results).mean()


def by_hour(results: pd.DataFrame) -> pd.DataFrame:
    """Return MAE and MAPE for each hour of the day (0 to 23)."""
    return _errors(results).groupby(results.index.hour).mean().rename_axis("hour")


def by_day_type(results: pd.DataFrame) -> pd.DataFrame:
    """Return MAE and MAPE for workdays, Saturdays, and Sundays or holidays."""
    days = results.index.normalize()
    types = day_types(days.unique()).reindex(days).to_numpy()
    return _errors(results).groupby(types).mean().reindex(DAY_TYPES).rename_axis("day_type")
