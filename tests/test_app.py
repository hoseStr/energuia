from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from demand_forecast import config
from demand_forecast.data import store, transform
from demand_forecast.models.backtest import History
from demand_forecast.models.registry import RegistryError, register_model
from demand_forecast.models.train import LightGBMForecaster
from demand_forecast.serving.app import create_app
from demand_forecast.serving.forecast import issue_forecast

BOGOTA = config.TIMEZONE
NOW = pd.Timestamp("2026-11-14 18:08", tz=BOGOTA)
VERSION = "20261113T163005Z"
BACKTEST = {
    "start": "2025-11-01",
    "end": "2026-10-31",
    "folds": 12,
    "metrics": {
        "seasonal-naive": {"mae": 338628.444, "mape": 3.4025},
        "moving-average": {"mae": 300503.5935, "mape": 3.034},
        "lightgbm": {"mae": 194519.7729, "mape": 1.9585},
    },
}


class Clock:
    """A clock the tests can move."""

    def __init__(self, now: pd.Timestamp) -> None:
        self.now = now

    def __call__(self) -> pd.Timestamp:
        return self.now


def write_data(data_dir: Path, make_wide) -> History:
    """Store synthetic demand up to 2026-11-08 and generation up to 2026-11-10."""
    for name, days in ((config.DEMAND.name, 496), (config.GENERATION.name, 498)):
        table = transform.hourly_table(make_wide("2025-07-01", days))
        store.write_table(table, store.processed_dir(name, data_dir), "timestamp")
    return History(
        demand=store.load_series(config.DEMAND.name, data_dir),
        generation=store.load_series(config.GENERATION.name, data_dir),
    )


def write_model(models_dir: Path, history: History) -> LightGBMForecaster:
    model = LightGBMForecaster(rounds=10, train_years=1)
    model.fit(history)
    register_model(
        model,
        trained_at=datetime(2026, 11, 13, 16, 30, 5, tzinfo=UTC),
        data={"fingerprint": "0123456789abcdef"},
        backtest=BACKTEST,
        commit={"commit": "abc123", "dirty": False},
        models_dir=models_dir,
    )
    return model


@pytest.fixture
def clock() -> Clock:
    return Clock(NOW)


@pytest.fixture
def history(tmp_path, make_wide) -> History:
    return write_data(tmp_path / "data", make_wide)


@pytest.fixture
def model(tmp_path, history) -> LightGBMForecaster:
    return write_model(tmp_path / "models", history)


@pytest.fixture
def client(tmp_path, model, clock) -> TestClient:
    return TestClient(create_app(tmp_path / "data", tmp_path / "models", clock=clock))


def test_health_says_which_model_and_data_the_service_holds(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_version": VERSION,
        "demand_last_day": "2026-11-08",
        "generation_last_day": "2026-11-10",
    }


def test_forecast_returns_the_24_hours_of_tomorrow(client, model, history):
    response = client.get("/forecast")

    assert response.status_code == 200
    body = response.json()
    hours = body.pop("hours")
    assert body == {
        "target_day": "2026-11-15",
        "issued_at": "2026-11-14T18:08:00-05:00",
        "model_version": VERSION,
        "demand_last_day": "2026-11-08",
        "generation_last_day": "2026-11-10",
        "timezone": BOGOTA,
        "unit": "kWh",
    }
    assert len(hours) == 24
    assert hours[0]["timestamp"] == "2026-11-15T00:00:00-05:00"
    assert hours[23]["timestamp"] == "2026-11-15T23:00:00-05:00"
    expected = issue_forecast(model, VERSION, history, now=NOW).values
    assert [hour["demand_kwh"] for hour in hours] == pytest.approx(list(expected))


def test_forecast_follows_the_clock(client, clock):
    clock.now = pd.Timestamp("2026-11-15 00:30", tz=BOGOTA)

    body = client.get("/forecast").json()

    assert body["target_day"] == "2026-11-16"
    assert body["issued_at"] == "2026-11-15T00:30:00-05:00"
    assert body["hours"][0]["timestamp"] == "2026-11-16T00:00:00-05:00"


def test_forecast_uses_the_bogota_date_when_the_server_clock_is_utc(client, clock):
    clock.now = pd.Timestamp("2026-11-15 02:00", tz="UTC")

    body = client.get("/forecast").json()

    assert body["target_day"] == "2026-11-15"
    assert body["issued_at"] == "2026-11-14T21:00:00-05:00"


def test_forecast_is_unavailable_when_the_data_cannot_back_it(client, clock):
    clock.now = pd.Timestamp("2026-11-10 09:00", tz=BOGOTA)

    response = client.get("/forecast")

    assert response.status_code == 503
    assert "2026-11-10 was not over" in response.json()["detail"]


def test_metrics_returns_the_backtest_behind_the_current_model(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.json() == {
        "model_version": VERSION,
        "backtest": {
            "start": "2025-11-01",
            "end": "2026-10-31",
            "folds": 12,
            "models": [
                {"model": "seasonal-naive", "mae_kwh": 338628.444, "mape_percent": 3.4025},
                {"model": "moving-average", "mae_kwh": 300503.5935, "mape_percent": 3.034},
                {"model": "lightgbm", "mae_kwh": 194519.7729, "mape_percent": 1.9585},
            ],
        },
    }


def test_the_api_documents_its_three_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {"/health", "/forecast", "/metrics"}


def test_the_bare_address_leads_to_the_documentation(client):
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"
    assert client.get("/docs").status_code == 200


def test_the_service_does_not_start_without_a_model(tmp_path, history):
    with pytest.raises(RegistryError, match="run the `train` command first"):
        create_app(tmp_path / "data", tmp_path / "models")


def test_the_service_does_not_start_without_data(tmp_path, make_wide):
    elsewhere = tmp_path / "elsewhere"
    write_model(tmp_path / "models", write_data(elsewhere, make_wide))

    with pytest.raises(FileNotFoundError, match="run the `ingest` command first"):
        create_app(tmp_path / "data", tmp_path / "models")
