"""Drift of what the model sees: the features of the logged forecasts against its training.

Only the load ratios are compared. Calendar features (hour, weekday, day of the year,
holidays) differ between a few recent weeks and ten years of training by construction, so
they would always look drifted.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from demand_forecast import config
from demand_forecast.data import store
from demand_forecast.features.build import build_features, training_frame
from demand_forecast.models.backtest import History
from demand_forecast.models.registry import registry_entry
from demand_forecast.serving.log import read_log

DRIFT_FEATURES = [
    "profile_shape",
    "same_weekday_last",
    "same_weekday_mean",
    "week_level",
    "demand_last_day_dev",
    "demand_last_hour_dev",
    "generation_last_day_dev",
    "generation_last_hour_dev",
    "generation_prev_day_dev",
    "generation_prev_hour_dev",
]
REPORT_FILE = "drift.html"
SUMMARY_FILE = "drift.json"
WEEK_HOURS = 7 * 24
# Smallest standard deviation the library divides by when it norms the distance.
MIN_SCALE = 0.001


@dataclass(frozen=True)
class DriftSummary:
    """Outcome of one drift report: a score per feature and which ones crossed the threshold."""

    model_version: str
    reference_start: str
    reference_end: str
    current_start: str
    current_end: str
    current_days: int
    method: str
    quantile: float
    thresholds: dict[str, float]
    scores: dict[str, float]
    drifted: list[str]


def reference_features(history: History, entry: dict[str, object]) -> pd.DataFrame:
    """Return the drift features of the days the registered model was trained on."""
    data, settings = entry["data"], entry["model"]
    first = pd.Timestamp(data["first_target_day"], tz=config.TIMEZONE)
    after_last = pd.Timestamp(data["last_target_day"], tz=config.TIMEZONE) + pd.Timedelta(days=1)
    trained_on = History(
        demand=history.demand[history.demand.index < after_last], generation=history.generation
    )
    frame = training_frame(
        trained_on,
        horizon_days=settings["horizon_days"],
        lag_days=settings["lag_days"],
        generation_lag_days=settings["generation_lag_days"],
        first_target=first,
    )
    return frame[DRIFT_FEATURES]


def current_features(
    history: History, log: pd.DataFrame, days: int = config.DRIFT_WINDOW_DAYS
) -> pd.DataFrame:
    """Return the drift features of the last `days` logged forecasts.

    Each day is rebuilt with the cuts written in the log, so these are the features the model
    saw when it issued that forecast (up to later revisions of the data by XM).
    """
    per_day = log.drop_duplicates("target_day").tail(days)
    if per_day.empty:
        return pd.DataFrame(columns=DRIFT_FEATURES, dtype=float)
    frame = build_features(
        history.demand,
        history.generation,
        pd.DatetimeIndex(per_day["target_day"]),
        demand_cut=pd.DatetimeIndex(per_day["demand_last_day"]),
        generation_cut=pd.DatetimeIndex(per_day["generation_last_day"]),
    )
    return frame[DRIFT_FEATURES]


def window_thresholds(
    reference: pd.DataFrame,
    hours: int,
    *,
    quantile: float = config.DRIFT_QUANTILE,
    step_hours: int = WEEK_HOURS,
) -> dict[str, float]:
    """Return, per feature, the drift score that ordinary windows of `hours` rarely reach.

    The score of every window of that length inside `reference`, taken every `step_hours`,
    is computed against the whole reference; the threshold is their `quantile`. With the
    default threshold of the library almost every ordinary month would count as drifted
    (explore/11_drift_threshold.py). Needs the `monitoring` dependency group.
    """
    from scipy.stats import wasserstein_distance

    starts = range(0, len(reference) - hours + 1, step_hours)
    if not starts:
        raise ValueError(f"the reference is shorter than the {hours} hours being compared")
    thresholds = {}
    for name in reference.columns:
        values = reference[name].to_numpy()
        scale = max(values.std(), MIN_SCALE)
        scores = [
            wasserstein_distance(values, values[start : start + hours]) / scale for start in starts
        ]
        thresholds[name] = float(np.quantile(scores, quantile))
    return thresholds


def drift_scores(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    html_path: Path,
    thresholds: dict[str, float],
) -> dict[str, float]:
    """Write Evidently's drift report to `html_path`; return the score of every feature.

    The score is the Wasserstein distance between both samples, in standard deviations of
    the reference. Needs the `monitoring` dependency group.
    """
    from evidently import Report
    from evidently.presets import DataDriftPreset

    preset = DataDriftPreset(num_method="wasserstein", per_column_threshold=thresholds)
    snapshot = Report([preset]).run(
        current_data=current.reset_index(drop=True), reference_data=reference.reset_index(drop=True)
    )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html_path))
    return {
        metric["config"]["column"]: float(metric["value"])
        for metric in snapshot.dict()["metrics"]
        if metric["config"]["type"].endswith(":ValueDrift")
    }


def write_drift(
    data_dir: Path = config.DATA_DIR,
    models_dir: Path = config.MODELS_DIR,
    site_dir: Path = config.SITE_DIR,
) -> DriftSummary | None:
    """Build the drift report of the logged forecasts into `site_dir`; None if nothing is logged."""
    history = History(
        demand=store.load_series(config.DEMAND.name, data_dir),
        generation=store.load_series(config.GENERATION.name, data_dir),
    )
    current = current_features(history, read_log(data_dir))
    if current.empty:
        return None
    entry = registry_entry(models_dir)
    reference = reference_features(history, entry)
    thresholds = window_thresholds(reference, len(current))
    scores = drift_scores(reference, current, site_dir / REPORT_FILE, thresholds)
    summary = DriftSummary(
        model_version=entry["version"],
        reference_start=reference.index[0].date().isoformat(),
        reference_end=reference.index[-1].date().isoformat(),
        current_start=current.index[0].date().isoformat(),
        current_end=current.index[-1].date().isoformat(),
        current_days=len(current) // 24,
        method="Wasserstein distance (normed)",
        quantile=config.DRIFT_QUANTILE,
        thresholds={name: round(value, 4) for name, value in thresholds.items()},
        scores={name: round(score, 4) for name, score in scores.items()},
        drifted=[name for name, score in scores.items() if score >= thresholds[name]],
    )
    (site_dir / SUMMARY_FILE).write_text(
        json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return summary


def read_drift(site_dir: Path = config.SITE_DIR) -> DriftSummary | None:
    """Return the summary of the last drift report built into `site_dir`, if there is one."""
    path = site_dir / SUMMARY_FILE
    if not path.exists():
        return None
    return DriftSummary(**json.loads(path.read_text(encoding="utf-8")))
