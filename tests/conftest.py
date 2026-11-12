"""Shared fixtures. Nothing here touches the network."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demand_forecast.data.schema import DATE_DTYPE, HOUR_COLUMNS
from demand_forecast.data.transform import wide_to_long

FIXTURES = Path(__file__).parent / "fixtures"


def _read_wide(name: str) -> pd.DataFrame:
    frame = pd.read_csv(FIXTURES / f"{name}_2026-10-01_2026-11-07.csv")
    frame["date"] = pd.to_datetime(frame["date"]).astype(DATE_DTYPE)
    return frame


@pytest.fixture
def demand_wide() -> pd.DataFrame:
    """Real DemaCome, 2026-10-01..2026-11-07, as downloaded on 2026-11-10.

    The last two days were still partial. 2026-11-01 is a Sunday and 2026-11-02 a holiday.
    """
    return _read_wide("demand")


@pytest.fixture
def generation_wide() -> pd.DataFrame:
    """Real Gene for the same days and download. Every day is complete."""
    return _read_wide("generation")


@pytest.fixture
def make_wide() -> Callable[..., pd.DataFrame]:
    """Build a synthetic wide table with a weekly and a daily pattern, and no noise."""

    def build(start: str, days: int, level: float = 1e7, growth: float = 1e-4) -> pd.DataFrame:
        dates = pd.date_range(start, periods=days, freq="D").astype(DATE_DTYPE)
        daily_shape = 1 + 0.2 * np.sin(np.arange(24) / 24 * 2 * np.pi)
        weekly_shape = np.where(dates.dayofweek == 6, 0.85, 1.0)
        trend = 1 + np.arange(days) * growth
        values = level * (weekly_shape * trend)[:, None] * daily_shape[None, :]
        frame = pd.DataFrame(values, columns=HOUR_COLUMNS)
        frame.insert(0, "date", dates)
        return frame

    return build


@pytest.fixture
def make_series(make_wide) -> Callable[..., pd.Series]:
    """Build a synthetic hourly series indexed by Bogota timestamps."""

    def build(start: str, days: int, **kwargs) -> pd.Series:
        return wide_to_long(make_wide(start, days, **kwargs)).set_index("timestamp")["value"]

    return build
