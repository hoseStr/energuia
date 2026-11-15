"""The public panel: one static page built from the forecast log and its measured error."""

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from demand_forecast import config
from demand_forecast.data import store
from demand_forecast.daytypes import day_types
from demand_forecast.models.registry import registry_entry
from demand_forecast.models.train import LightGBMForecaster
from demand_forecast.monitoring.drift import DriftSummary, read_drift
from demand_forecast.monitoring.evaluate import evaluate, recent_mape
from demand_forecast.serving.log import latest_forecast, read_log

TEMPLATE = Path(__file__).with_name("panel.html")
PLACEHOLDER = "__PAYLOAD__"
PANEL_FILE = "index.html"
KWH_PER_GWH = 1e6


def build_payload(
    log: pd.DataFrame,
    demand: pd.Series,
    entry: dict[str, object],
    *,
    now: pd.Timestamp,
    drift: DriftSummary | None = None,
    days: int = config.PANEL_DAYS,
) -> dict[str, object]:
    """Return everything the panel shows, ready to be serialised.

    `log` is the forecast log, `demand` the processed demand and `entry` the registry entry of
    the current model. The hourly series covers the last `days` logged days.
    """
    daily = evaluate(log, demand)
    scores = entry["backtest"]["metrics"]
    backtest_mape = scores[LightGBMForecaster.name]["mape"]
    baselines = [score["mape"] for name, score in scores.items() if name != LightGBMForecaster.name]

    live = None
    if not daily.empty:
        live = {
            "evaluated_days": len(daily),
            "first_day": daily["target_day"].iloc[0].date().isoformat(),
            "last_day": daily["target_day"].iloc[-1].date().isoformat(),
            "mape": round(float(daily["mape_percent"].mean()), 3),
            "recent_days": config.ALERT_WINDOW_DAYS,
            "recent_mape": None if (recent := recent_mape(daily)) is None else round(recent, 3),
        }

    latest = latest_forecast(log)
    logged_days = log.drop_duplicates("target_day")["target_day"]
    recent_hours = log[log["target_day"].isin(logged_days.tail(days))]
    actual = demand.reindex(pd.DatetimeIndex(recent_hours["timestamp"])).to_numpy()
    hourly = [
        [
            timestamp.isoformat(),
            round(predicted / KWH_PER_GWH, 4),
            None if pd.isna(value) else round(value / KWH_PER_GWH, 4),
        ]
        for timestamp, predicted, value in zip(
            recent_hours["timestamp"], recent_hours["predicted_kwh"], actual, strict=True
        )
    ]

    totals = log.groupby("target_day", sort=True)["predicted_kwh"].sum()
    scored = daily.set_index("target_day")
    running = scored["mape_percent"].expanding().mean()
    types = day_types(pd.DatetimeIndex(totals.index))
    table = []
    for target_day, predicted in totals.items():
        evaluated = target_day in scored.index
        table.append(
            {
                "day": target_day.date().isoformat(),
                "day_type": types[target_day],
                "predicted_gwh": round(predicted / KWH_PER_GWH, 2),
                "actual_gwh": round(scored.at[target_day, "actual_kwh"] / KWH_PER_GWH, 2)
                if evaluated
                else None,
                "mape": round(float(scored.at[target_day, "mape_percent"]), 3)
                if evaluated
                else None,
                "cumulative_mape": round(float(running[target_day]), 3) if evaluated else None,
            }
        )

    return {
        "generated_at": now.tz_convert(config.TIMEZONE).isoformat(timespec="seconds"),
        "timezone": config.TIMEZONE,
        "model_version": entry["version"],
        "backtest": {
            "start": entry["backtest"]["start"],
            "end": entry["backtest"]["end"],
            "mape": round(backtest_mape, 3),
            "best_baseline_mape": round(min(baselines), 3),
        },
        "recent_days": config.ALERT_WINDOW_DAYS,
        "alert_threshold": round(config.ALERT_MAPE_FACTOR * backtest_mape, 3),
        "live": live,
        "latest": None
        if latest is None
        else {
            "target_day": latest.target_day.date().isoformat(),
            "issued_at": latest.issued_at.isoformat(timespec="seconds"),
            "total_gwh": round(float(latest.values.sum()) / KWH_PER_GWH, 2),
        },
        "hourly": hourly,
        "days": table,
        "drift": None if drift is None else asdict(drift),
        "drift_min_days": config.DRIFT_MIN_DAYS,
        "links": {"repo": config.REPO_URL, "api": config.API_URL, "log": config.LOG_URL},
    }


def render(payload: dict[str, object]) -> str:
    """Return the page with `payload` embedded as JSON."""
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.read_text(encoding="utf-8").replace(PLACEHOLDER, data)


def write_panel(
    data_dir: Path = config.DATA_DIR,
    models_dir: Path = config.MODELS_DIR,
    site_dir: Path = config.SITE_DIR,
    *,
    now: pd.Timestamp,
) -> Path:
    """Build `index.html` into `site_dir`; return its path.

    It links to the drift report when one was already built into the same folder.
    """
    payload = build_payload(
        read_log(data_dir),
        store.load_series(config.DEMAND.name, data_dir),
        registry_entry(models_dir),
        now=now,
        drift=read_drift(site_dir),
    )
    site_dir.mkdir(parents=True, exist_ok=True)
    path = site_dir / PANEL_FILE
    path.write_text(render(payload), encoding="utf-8", newline="\n")
    return path
