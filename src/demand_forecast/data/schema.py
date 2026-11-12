"""Shapes of the stored tables, enforced with Pandera."""

import pandas as pd
import pandera.pandas as pa

from demand_forecast import config

HOUR_COLUMNS = [f"hour_{hour:02d}" for hour in range(1, 25)]
WIDE_COLUMNS = ["date", *HOUR_COLUMNS]
DATE_DTYPE = "datetime64[us]"
TIMESTAMP_DTYPE = pd.DatetimeTZDtype("us", config.TIMEZONE)
FETCHED_AT_DTYPE = pd.DatetimeTZDtype("us", "UTC")


def _whole_days_of_consecutive_hours(frame: pd.DataFrame) -> bool:
    timestamps = frame["timestamp"]
    if timestamps.empty:
        return True
    consecutive = timestamps.diff().dropna().eq(pd.Timedelta(hours=1)).all()
    return bool(consecutive and timestamps.iloc[0].hour == 0 and len(timestamps) % 24 == 0)


# One row per day, exactly as XM returned it. Partial days are valid here.
RAW_SCHEMA = pa.DataFrameSchema(
    {
        "date": pa.Column(DATE_DTYPE, unique=True),
        **{column: pa.Column(float, pa.Check.ge(0), nullable=True) for column in HOUR_COLUMNS},
        "fetched_at": pa.Column(FETCHED_AT_DTYPE),
    },
    checks=pa.Check(
        lambda frame: frame["date"].is_monotonic_increasing, error="dates must be sorted"
    ),
    strict=True,
    ordered=True,
)

# One row per hour, complete days only, with no gap between the first and the last hour.
HOURLY_SCHEMA = pa.DataFrameSchema(
    {
        "timestamp": pa.Column(TIMESTAMP_DTYPE, unique=True),
        "value": pa.Column(
            float, pa.Check.in_range(config.PLAUSIBLE_KWH_MIN, config.PLAUSIBLE_KWH_MAX)
        ),
        "anomaly": pa.Column(bool),
    },
    checks=pa.Check(
        _whole_days_of_consecutive_hours,
        error="timestamps must be consecutive hours covering whole days",
    ),
    strict=True,
    ordered=True,
)


def validate_raw(frame: pd.DataFrame) -> pd.DataFrame:
    """Return `frame` if it is a valid raw table; raise `SchemaErrors` otherwise."""
    return RAW_SCHEMA.validate(frame, lazy=True)


def validate_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    """Return `frame` if it is a valid hourly table; raise `SchemaErrors` otherwise."""
    return HOURLY_SCHEMA.validate(frame, lazy=True)
