"""Error of the logged forecasts against the demand XM published afterwards.

`data/metrics/daily.csv` is derived: every run rebuilds it from the forecast log and the
processed demand, so it always reflects the latest published values.
"""

from pathlib import Path

import pandas as pd

from demand_forecast import config
from demand_forecast.daytypes import day_types

HOURS_PER_DAY = 24
COLUMNS = [
    "target_day",
    "day_type",
    "model_version",
    "actual_kwh",
    "predicted_kwh",
    "mae_kwh",
    "mape_percent",
]
METRICS_FILE = "daily.csv"


def evaluate(log: pd.DataFrame, demand: pd.Series) -> pd.DataFrame:
    """Return the error of every logged day whose 24 hours of demand are published.

    One row per day: the daily totals in kWh, the mean absolute error of its hours and their
    mean absolute percentage error.
    """
    hours = log[log["timestamp"].isin(demand.index)].copy()
    if hours.empty:
        return pd.DataFrame({name: [] for name in COLUMNS})
    hours["actual"] = demand.loc[hours["timestamp"]].to_numpy()
    hours["error"] = (hours["predicted_kwh"] - hours["actual"]).abs()
    hours["percent"] = hours["error"] / hours["actual"] * 100
    days = hours.groupby("target_day", sort=True)
    daily = days.agg(
        model_version=("model_version", "first"),
        actual_kwh=("actual", "sum"),
        predicted_kwh=("predicted_kwh", "sum"),
        mae_kwh=("error", "mean"),
        mape_percent=("percent", "mean"),
        hours=("actual", "size"),
    )
    daily = daily[daily["hours"] == HOURS_PER_DAY].drop(columns="hours").reset_index()
    daily["day_type"] = day_types(pd.DatetimeIndex(daily["target_day"])).to_numpy()
    return daily[COLUMNS]


def metrics_path(data_dir: Path = config.DATA_DIR) -> Path:
    """File with the error of every evaluated day."""
    return data_dir / "metrics" / METRICS_FILE


def write_metrics(daily: pd.DataFrame, data_dir: Path = config.DATA_DIR) -> Path:
    """Replace the metrics file with `daily`; return its path."""
    table = daily.copy()
    table["target_day"] = pd.DatetimeIndex(table["target_day"]).strftime("%Y-%m-%d")
    for name in ("actual_kwh", "predicted_kwh", "mae_kwh"):
        table[name] = table[name].map("{:.3f}".format)
    table["mape_percent"] = table["mape_percent"].map("{:.4f}".format)
    path = metrics_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        table[COLUMNS].to_csv(file, index=False, lineterminator="\n")
    return path


def read_metrics(data_dir: Path = config.DATA_DIR) -> pd.DataFrame:
    """Return the stored error of every evaluated day; empty when nothing was evaluated yet."""
    path = metrics_path(data_dir)
    if not path.exists():
        return pd.DataFrame({name: [] for name in COLUMNS})
    daily = pd.read_csv(path, dtype={"model_version": str, "day_type": str})
    daily["target_day"] = pd.to_datetime(daily["target_day"]).dt.tz_localize(config.TIMEZONE)
    return daily[COLUMNS]


def recent_mape(daily: pd.DataFrame, days: int = config.ALERT_WINDOW_DAYS) -> float | None:
    """Return the MAPE of the last `days` evaluated days, or None when there are fewer."""
    if len(daily) < days:
        return None
    return float(daily["mape_percent"].iloc[-days:].mean())


def alert_message(
    daily: pd.DataFrame,
    backtest_mape: float,
    *,
    days: int = config.ALERT_WINDOW_DAYS,
    factor: float = config.ALERT_MAPE_FACTOR,
) -> str | None:
    """Return a warning when the recent error exceeds `factor` times the backtest error."""
    recent = recent_mape(daily, days)
    threshold = factor * backtest_mape
    if recent is None or recent <= threshold:
        return None
    window = pd.DatetimeIndex(daily["target_day"].iloc[-days:])
    return (
        f"The MAPE of the last {days} evaluated days ({window[0].date()} to "
        f"{window[-1].date()}) is {recent:.2f} %, above the alert threshold of "
        f"{threshold:.2f} % ({factor:g} times the {backtest_mape:.2f} % of the backtest)."
    )
