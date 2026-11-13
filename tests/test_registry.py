import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from demand_forecast import config
from demand_forecast.features.build import FEATURES
from demand_forecast.models.backtest import history_at
from demand_forecast.models.registry import (
    RegistryError,
    git_commit,
    load_model,
    read_registry,
    register_model,
)
from demand_forecast.models.train import LightGBMForecaster

BOGOTA = config.TIMEZONE
TRAINED_AT = datetime(2026, 11, 13, 16, 30, 5, tzinfo=UTC)
COMMIT = {"commit": "abc123", "dirty": False}
DATA = {"fingerprint": "0123456789abcdef"}
BACKTEST = {"start": "2025-11-01", "end": "2026-10-31", "metrics": {"lightgbm": {"mape": 2.0}}}


def day(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=BOGOTA)


@pytest.fixture
def series(make_series) -> pd.Series:
    rng = np.random.default_rng(5)
    clean = make_series("2025-06-01", 300)
    return clean * rng.normal(1, 0.03, len(clean))


@pytest.fixture
def fitted(series) -> LightGBMForecaster:
    model = LightGBMForecaster(rounds=20, train_years=2)
    model.fit(history_at(series, series, day("2026-02-01")))
    return model


def register(model, models_dir, trained_at=TRAINED_AT) -> dict:
    return register_model(
        model,
        trained_at=trained_at,
        data=DATA,
        backtest=BACKTEST,
        commit=COMMIT,
        models_dir=models_dir,
    )


def test_registering_writes_the_model_and_its_metadata(fitted, tmp_path):
    entry = register(fitted, tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "lightgbm-20261113T163005Z.txt",
        "registry.json",
    ]
    registry = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))
    assert registry == {"current": "20261113T163005Z", "models": [entry]}
    assert entry["trained_at"] == "2026-11-13T16:30:05+00:00"
    assert entry["commit"] == "abc123"
    assert entry["dirty"] is False
    assert entry["data"] == {
        "first_target_day": "2025-07-04",
        "last_target_day": "2026-01-27",
        "rows": 208 * 24,
        "fingerprint": "0123456789abcdef",
    }
    assert entry["model"]["rounds"] == 20
    assert entry["model"]["train_years"] == 2
    assert entry["model"]["lag_days"] == config.DATA_LAG_DAYS
    assert entry["model"]["generation_lag_days"] == config.GENERATION_LAG_DAYS
    assert entry["model"]["features"] == FEATURES
    assert entry["backtest"] == BACKTEST


def test_a_loaded_model_forecasts_exactly_like_the_one_that_was_saved(fitted, series, tmp_path):
    register(fitted, tmp_path)
    loaded, entry = load_model(tmp_path)

    history = history_at(series, series, day("2026-02-09"))
    expected = fitted.predict(history, day("2026-02-10"))
    assert list(loaded.predict(history, day("2026-02-10"))) == list(expected)
    assert entry["version"] == "20261113T163005Z"
    assert loaded.rounds == 20
    assert loaded.train_years == 2


def test_the_last_registered_model_is_the_current_one(fitted, tmp_path):
    register(fitted, tmp_path)
    later = register(fitted, tmp_path, trained_at=datetime(2026, 11, 20, 16, 0, tzinfo=UTC))

    registry = read_registry(tmp_path)
    assert registry["current"] == later["version"]
    assert [entry["version"] for entry in registry["models"]] == [
        "20261113T163005Z",
        "20261120T160000Z",
    ]
    assert load_model(tmp_path)[1]["version"] == later["version"]
    assert load_model(tmp_path, "20261113T163005Z")[1]["version"] == "20261113T163005Z"


def test_the_same_version_cannot_be_registered_twice(fitted, tmp_path):
    register(fitted, tmp_path)
    with pytest.raises(RegistryError, match="already registered"):
        register(fitted, tmp_path)


def test_an_unfitted_model_cannot_be_registered(tmp_path):
    with pytest.raises(RegistryError, match="fitted"):
        register(LightGBMForecaster(), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_an_empty_registry_says_how_to_get_a_model(tmp_path):
    assert read_registry(tmp_path) == {"current": None, "models": []}
    with pytest.raises(RegistryError, match="run the `train` command first"):
        load_model(tmp_path)


def test_an_unknown_version_is_an_error(fitted, tmp_path):
    register(fitted, tmp_path)
    with pytest.raises(RegistryError, match="20200101T000000Z is not in the registry"):
        load_model(tmp_path, "20200101T000000Z")


def test_a_tampered_model_file_is_rejected(fitted, tmp_path):
    entry = register(fitted, tmp_path)
    path = tmp_path / entry["file"]
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(RegistryError, match="does not match the hash"):
        load_model(tmp_path)


def test_a_missing_model_file_is_an_error(fitted, tmp_path):
    entry = register(fitted, tmp_path)
    (tmp_path / entry["file"]).unlink()
    with pytest.raises(RegistryError, match="is missing"):
        load_model(tmp_path)


def test_a_model_trained_with_other_features_is_rejected(fitted, tmp_path):
    register(fitted, tmp_path)
    path = tmp_path / "registry.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    registry["models"][0]["model"]["features"] = FEATURES[:-1]
    path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(RegistryError, match="other features"):
        load_model(tmp_path)


def test_git_commit_outside_a_repository_is_unknown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert git_commit() == {"commit": None, "dirty": None}
