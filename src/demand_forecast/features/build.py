"""Features of the day-ahead forecast, built only from what is published when it is issued.

Every row describes one target hour. The caller says, for each target day, which is the last
published day of each series (its cut); nothing after a cut is ever read. Load features are
ratios, never kWh, because the level of the demand doubled between 2000 and 2025.

The reference for those ratios is the profile: the mean, hour by hour, of the days of the
window that have the day type of the target. The model forecasts the demand relative to it.
"""

import numpy as np
import pandas as pd

from demand_forecast import config
from demand_forecast.daytypes import DAY_TYPES, day_type_codes, is_holiday
from demand_forecast.models.backtest import History

HOURS_PER_DAY = 24
ONE_DAY = pd.Timedelta(days=1)

CALENDAR_FEATURES = [
    "hour",
    "weekday",
    "day_type",
    "is_holiday",
    "holiday_eve",
    "after_holiday",
    "day_of_year",
]
WINDOW_FEATURES = [
    "profile_shape",
    "same_type_days",
    "same_weekday_last",
    "same_weekday_mean",
    "same_weekday_mismatch",
    "week_level",
    "window_holidays",
]
RECENT_FEATURES = [
    "demand_last_day_dev",
    "demand_last_hour_dev",
    "generation_last_day_dev",
    "generation_last_hour_dev",
    "generation_prev_day_dev",
    "generation_prev_hour_dev",
]
FEATURES = [*CALENDAR_FEATURES, *WINDOW_FEATURES, *RECENT_FEATURES]
# kWh of the profile at the target hour: forecast = model output * scale.
SCALE = "scale"
# Demand of the target hour relative to the scale. Only training frames have it.
TARGET = "target"


def _daily(series: pd.Series, name: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Return the days of an hourly series and its values as a (days, 24) matrix."""
    index = series.index
    whole_days = len(index) % HOURS_PER_DAY == 0 and (len(index) == 0 or index[0].hour == 0)
    if not whole_days or index[-1] - index[0] != pd.Timedelta(hours=len(index) - 1):
        raise ValueError(f"the {name} history must be hourly, without gaps, in whole days")
    return index[::HOURS_PER_DAY], series.to_numpy(dtype=float).reshape(-1, HOURS_PER_DAY)


def _positions(days: pd.DatetimeIndex, wanted: pd.DatetimeIndex, name: str) -> np.ndarray:
    positions = days.get_indexer(wanted)
    if (positions < 0).any():
        missing = wanted[positions < 0][0].date()
        raise ValueError(f"the {name} history does not have {missing}")
    return positions


def _masked_mean(window: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean over the days of each window that `mask` selects, hour by hour. NaN if none."""
    counts = mask.sum(axis=1, keepdims=True).astype(float)
    counts[counts == 0] = np.nan
    return np.where(mask[:, :, None], window, 0.0).sum(axis=1) / counts


def build_features(
    demand: pd.Series,
    generation: pd.Series,
    target_days: pd.DatetimeIndex,
    *,
    demand_cut: pd.DatetimeIndex,
    generation_cut: pd.DatetimeIndex,
    window_days: int = config.FEATURE_WINDOW_DAYS,
) -> pd.DataFrame:
    """Return one row per hour of every target day, with `FEATURES` and `SCALE`.

    `demand_cut` and `generation_cut` give, for each target day, the last day of each series
    published when its forecast is issued. Values after a cut are never read.
    """
    target_days = pd.DatetimeIndex(target_days)
    demand_cut, generation_cut = pd.DatetimeIndex(demand_cut), pd.DatetimeIndex(generation_cut)
    if (demand_cut >= target_days).any() or (generation_cut >= target_days).any():
        raise ValueError("every cut must fall before its target day")

    demand_days, demand_values = _daily(demand, "demand")
    generation_days, generation_values = _daily(generation, "generation")
    last = _positions(demand_days, demand_cut, "demand")
    generation_last = _positions(generation_days, generation_cut, "generation")
    generation_prev = _positions(generation_days, generation_cut - ONE_DAY, "generation")
    if (last < window_days - 1).any():
        raise ValueError(f"the features need {window_days} days of demand up to the cut")

    each = np.arange(len(target_days))
    in_window = last[:, None] + np.arange(1 - window_days, 1)
    window = demand_values[in_window]
    window_types = day_type_codes(demand_days)[in_window]
    target_types = day_type_codes(target_days)

    level = window.mean(axis=(1, 2))
    profiles = np.stack(
        [_masked_mean(window, window_types == code) for code in range(len(DAY_TYPES))], axis=1
    )
    profile = profiles[each, target_types]
    if np.isnan(profile).any():
        raise ValueError("a window has no day with the day type of its target")

    same_weekday = (
        demand_days.dayofweek.to_numpy()[in_window] == target_days.dayofweek.to_numpy()[:, None]
    )
    last_same_weekday = window_days - 1 - np.argmax(same_weekday[:, ::-1], axis=1)

    def deviation(values: np.ndarray, days: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        """How far a day sits from the profile of its own day type: as a whole, and by hour."""
        expected = profiles[each, day_type_codes(days)]
        return values.mean(axis=1) / expected.mean(axis=1), values / expected

    demand_day_dev, demand_hour_dev = deviation(demand_values[last], demand_cut)
    last_day_dev, last_hour_dev = deviation(generation_values[generation_last], generation_cut)
    prev_day_dev, prev_hour_dev = deviation(
        generation_values[generation_prev], generation_cut - ONE_DAY
    )

    by_day = {
        "weekday": target_days.dayofweek.to_numpy(),
        "day_type": target_types,
        "is_holiday": is_holiday(target_days).astype(int),
        "holiday_eve": is_holiday(target_days + ONE_DAY).astype(int),
        "after_holiday": is_holiday(target_days - ONE_DAY).astype(int),
        "day_of_year": target_days.dayofyear.to_numpy(),
        "same_type_days": (window_types == target_types[:, None]).sum(axis=1),
        "same_weekday_mismatch": (same_weekday & (window_types != target_types[:, None])).sum(
            axis=1
        ),
        "week_level": window[:, -7:].mean(axis=(1, 2)) / level,
        "window_holidays": is_holiday(demand_days)[in_window].sum(axis=1),
        "demand_last_day_dev": demand_day_dev,
        "generation_last_day_dev": last_day_dev,
        "generation_prev_day_dev": prev_day_dev,
    }
    by_hour = {
        "profile_shape": profile / level[:, None],
        "same_weekday_last": window[each, last_same_weekday] / profile,
        "same_weekday_mean": _masked_mean(window, same_weekday) / profile,
        "demand_last_hour_dev": demand_hour_dev,
        "generation_last_hour_dev": last_hour_dev,
        "generation_prev_hour_dev": prev_hour_dev,
        SCALE: profile,
    }
    hours = np.tile(np.arange(HOURS_PER_DAY), len(target_days))
    columns = {"hour": hours}
    columns |= {name: np.repeat(values, HOURS_PER_DAY) for name, values in by_day.items()}
    columns |= {name: values.ravel() for name, values in by_hour.items()}
    timestamps = target_days.repeat(HOURS_PER_DAY) + pd.to_timedelta(hours, unit="h")
    return pd.DataFrame(columns, index=timestamps)[[*FEATURES, SCALE]]


def training_frame(
    history: History,
    *,
    horizon_days: int = config.HORIZON_DAYS,
    lag_days: int = config.DATA_LAG_DAYS,
    generation_lag_days: int = config.GENERATION_LAG_DAYS,
    first_target: pd.Timestamp | None = None,
    window_days: int = config.FEATURE_WINDOW_DAYS,
) -> pd.DataFrame:
    """Return `FEATURES`, `SCALE` and `TARGET` for every published day that can be a target.

    Each row is built as if that day had been forecast live: with the demand published
    `horizon_days + lag_days` days before it, and the generation `horizon_days +
    generation_lag_days` days before it.
    """
    if history.generation is None:
        raise ValueError("the features need the generation history")
    days, _ = _daily(history.demand, "demand")
    demand_gap = pd.Timedelta(days=horizon_days + lag_days)
    generation_gap = pd.Timedelta(days=horizon_days + generation_lag_days)
    target_days = days[days >= days[0] + pd.Timedelta(days=window_days - 1) + demand_gap]
    if first_target is not None:
        target_days = target_days[target_days >= first_target]
    frame = build_features(
        history.demand,
        history.generation,
        target_days,
        demand_cut=target_days - demand_gap,
        generation_cut=target_days - generation_gap,
        window_days=window_days,
    )
    frame[TARGET] = history.demand.loc[frame.index].to_numpy() / frame[SCALE].to_numpy()
    return frame
