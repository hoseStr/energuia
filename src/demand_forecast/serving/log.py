"""Log of issued forecasts: `data/predictions/<year>.csv`. Lines are added, never rewritten.

It is plain text so that the history of the repository shows, line by line, that every
forecast was written before its day and never touched again.
"""

from pathlib import Path

import pandas as pd

from demand_forecast import config
from demand_forecast.serving.forecast import Forecast

HOURS_PER_DAY = 24
STORED_COLUMNS = [
    "timestamp",
    "predicted_kwh",
    "issued_at",
    "model_version",
    "demand_last_day",
    "generation_last_day",
]
# `target_day` is not stored: it is the day of `timestamp`.
COLUMNS = [*STORED_COLUMNS, "target_day"]


class LogError(Exception):
    """The forecast cannot be added to the log. Nothing was written."""


def predictions_dir(data_dir: Path = config.DATA_DIR) -> Path:
    """Folder of the forecast log."""
    return data_dir / "predictions"


def read_log(data_dir: Path = config.DATA_DIR) -> pd.DataFrame:
    """Return every logged forecast, one row per hour, in the order they were issued."""
    files = sorted(predictions_dir(data_dir).glob("*.csv"))
    if not files:
        empty = pd.DataFrame({name: [] for name in COLUMNS})
        moments = ("timestamp", "issued_at", "target_day", "demand_last_day", "generation_last_day")
        for name in moments:
            empty[name] = pd.to_datetime(empty[name]).dt.tz_localize(config.TIMEZONE)
        return empty.astype({"predicted_kwh": float, "model_version": str})
    text_columns = {name: str for name in STORED_COLUMNS if name != "predicted_kwh"}
    log = pd.concat([pd.read_csv(file, dtype=text_columns) for file in files], ignore_index=True)
    for name in ("timestamp", "issued_at"):
        log[name] = pd.to_datetime(log[name], utc=True).dt.tz_convert(config.TIMEZONE)
    for name in ("demand_last_day", "generation_last_day"):
        log[name] = pd.to_datetime(log[name]).dt.tz_localize(config.TIMEZONE)
    log["target_day"] = log["timestamp"].dt.normalize()
    return log[COLUMNS]


def latest_forecast(log: pd.DataFrame) -> Forecast | None:
    """Return the forecast of the last logged day, or None when the log is empty."""
    if log.empty:
        return None
    rows = log[log["target_day"] == log["target_day"].iloc[-1]]
    first = rows.iloc[0]
    return Forecast(
        issued_at=first["issued_at"],
        target_day=first["target_day"],
        model_version=first["model_version"],
        demand_last_day=first["demand_last_day"],
        generation_last_day=first["generation_last_day"],
        values=pd.Series(
            rows["predicted_kwh"].to_numpy(), index=pd.DatetimeIndex(rows["timestamp"])
        ),
    )


def append_forecast(forecast: Forecast, data_dir: Path = config.DATA_DIR) -> Path:
    """Add the 24 hours of `forecast` at the end of the log; return the file that grew.

    Refuses a day that is already logged, a day before the last logged one, and a forecast
    issued once its day had started: the log only holds forecasts made ahead of time, in the
    order they were made.
    """
    target_day = forecast.target_day
    values = forecast.values
    expected = pd.date_range(target_day, periods=HOURS_PER_DAY, freq="h")
    if len(values) != HOURS_PER_DAY or not values.index.equals(expected):
        raise LogError(f"a forecast must hold the 24 hours of {target_day.date()}")
    if not (values > 0).all():
        raise LogError(f"the forecast for {target_day.date()} must be positive in every hour")
    if forecast.issued_at >= target_day:
        raise LogError(
            f"the forecast was issued after {target_day.date()} started "
            f"({forecast.issued_at.isoformat(timespec='seconds')})"
        )
    logged_days = read_log(data_dir)["target_day"]
    if not logged_days.empty:
        last_day = logged_days.iloc[-1]
        if (logged_days == target_day).any():
            raise LogError(f"{target_day.date()} is already in the log")
        if target_day < last_day:
            raise LogError(
                f"the log already reaches {last_day.date()}; {target_day.date()} cannot be "
                "added afterwards"
            )

    issued_at = forecast.issued_at.tz_convert(config.TIMEZONE).isoformat(timespec="seconds")
    tail = ",".join(
        [
            issued_at,
            forecast.model_version,
            forecast.demand_last_day.date().isoformat(),
            forecast.generation_last_day.date().isoformat(),
        ]
    )
    lines = [f"{hour.isoformat()},{value:.3f},{tail}" for hour, value in values.items()]
    path = predictions_dir(data_dir) / f"{target_day.year}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [] if path.exists() else [",".join(STORED_COLUMNS)]
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write("\n".join([*header, *lines]) + "\n")
    return path
