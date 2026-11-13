"""Model files and their metadata, kept in the repo: `models/registry.json`."""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb

from demand_forecast import config
from demand_forecast.features.build import FEATURES
from demand_forecast.models.train import LightGBMForecaster

REGISTRY_NAME = "registry.json"


class RegistryError(Exception):
    """The registry or a model file cannot be trusted."""


def git_commit() -> dict[str, object]:
    """Return the commit of the working tree and whether it has uncommitted changes."""

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        return done.stdout.strip()

    commit, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def read_registry(models_dir: Path = config.MODELS_DIR) -> dict[str, object]:
    """Return the registry, empty when no model has been registered yet."""
    path = models_dir / REGISTRY_NAME
    if not path.exists():
        return {"current": None, "models": []}
    return json.loads(path.read_text(encoding="utf-8"))


def register_model(
    forecaster: LightGBMForecaster,
    *,
    trained_at: datetime,
    data: dict[str, object],
    backtest: dict[str, object],
    commit: dict[str, object],
    models_dir: Path = config.MODELS_DIR,
) -> dict[str, object]:
    """Save a fitted model next to its metadata and make it the current one.

    `trained_at` must be timezone-aware. `data` describes the data version and `backtest` the
    evaluation that backs the model; both are stored as given.
    """
    if forecaster.booster is None or forecaster.training_data is None:
        raise RegistryError("only a fitted model can be registered")
    registry = read_registry(models_dir)
    version = trained_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    if any(entry["version"] == version for entry in registry["models"]):
        raise RegistryError(f"version {version} is already registered")

    text = forecaster.booster.model_to_string()
    file_name = f"{forecaster.name}-{version}.txt"
    trained_on = forecaster.training_data
    entry = {
        "version": version,
        "file": file_name,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "trained_at": trained_at.astimezone(UTC).isoformat(timespec="seconds"),
        **commit,
        "data": {
            "first_target_day": trained_on.first_target_day.date().isoformat(),
            "last_target_day": trained_on.last_target_day.date().isoformat(),
            "rows": trained_on.rows,
            **data,
        },
        "model": {
            "params": forecaster.params,
            "rounds": forecaster.rounds,
            "train_years": forecaster.train_years,
            "horizon_days": forecaster.horizon_days,
            "lag_days": forecaster.lag_days,
            "generation_lag_days": forecaster.generation_lag_days,
            "features": FEATURES,
        },
        "backtest": backtest,
    }
    registry["models"].append(entry)
    registry["current"] = version

    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / file_name).write_text(text, encoding="utf-8", newline="\n")
    temporary = models_dir / f"{REGISTRY_NAME}.tmp"
    temporary.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    temporary.replace(models_dir / REGISTRY_NAME)
    return entry


def load_model(
    models_dir: Path = config.MODELS_DIR, version: str | None = None
) -> tuple[LightGBMForecaster, dict[str, object]]:
    """Return a registered model, the current one by default, and its registry entry."""
    registry = read_registry(models_dir)
    version = version or registry["current"]
    if version is None:
        raise RegistryError(f"no model registered in {models_dir}; run the `train` command first")
    entries = [entry for entry in registry["models"] if entry["version"] == version]
    if not entries:
        raise RegistryError(f"version {version} is not in the registry")
    (entry,) = entries

    path = models_dir / entry["file"]
    if not path.exists():
        raise RegistryError(f"the model file {path} is missing")
    text = path.read_text(encoding="utf-8")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != entry["sha256"]:
        raise RegistryError(f"{path} does not match the hash in the registry")
    settings = entry["model"]
    if settings["features"] != FEATURES:
        raise RegistryError(f"version {version} was trained with other features than this code")
    forecaster = LightGBMForecaster(
        settings["params"],
        rounds=settings["rounds"],
        train_years=settings["train_years"],
        horizon_days=settings["horizon_days"],
        lag_days=settings["lag_days"],
        generation_lag_days=settings["generation_lag_days"],
        booster=lgb.Booster(model_str=text),
    )
    return forecaster, entry
