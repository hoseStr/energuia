import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from demand_forecast import config
from demand_forecast.models.registry import RegistryError
from demand_forecast.monitoring.evaluate import write_metrics
from demand_forecast.serving.app import create_app
from demand_forecast.serving.forecast import Forecast
from demand_forecast.serving.log import append_forecast

BOGOTA = config.TIMEZONE
CURRENT = "20261120T160000Z"
EARLIER = "20261113T163005Z"
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


def day(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=BOGOTA)


def write_registry(models_dir: Path) -> None:
    models_dir.mkdir(parents=True)
    entries = [{"version": version, "backtest": BACKTEST} for version in (EARLIER, CURRENT)]
    registry = {"current": CURRENT, "models": entries}
    (models_dir / "registry.json").write_text(json.dumps(registry), encoding="utf-8")


def log_forecast(data_dir: Path, target: str, issued: str, level: float) -> Forecast:
    target_day = day(target)
    forecast = Forecast(
        issued_at=day(issued),
        target_day=target_day,
        model_version=EARLIER,
        demand_last_day=target_day - pd.Timedelta(days=7),
        generation_last_day=target_day - pd.Timedelta(days=5),
        values=pd.Series(
            level + np.arange(24) * 1000.0, index=pd.date_range(target_day, periods=24, freq="h")
        ),
    )
    append_forecast(forecast, data_dir)
    return forecast


def write_evaluated_days(data_dir: Path, mapes: list[float]) -> None:
    daily = pd.DataFrame(
        {
            "target_day": pd.date_range(day("2026-11-15"), periods=len(mapes), freq="D"),
            "day_type": "workday",
            "model_version": EARLIER,
            "actual_kwh": 2.4e8,
            "predicted_kwh": 2.4e8,
            "mae_kwh": [100000.0 * mape for mape in mapes],
            "mape_percent": mapes,
        }
    )
    write_metrics(daily, data_dir)


@pytest.fixture
def dirs(tmp_path) -> tuple[Path, Path]:
    write_registry(tmp_path / "models")
    return tmp_path / "data", tmp_path / "models"


def client_of(dirs) -> TestClient:
    return TestClient(create_app(*dirs))


def test_forecast_returns_the_latest_logged_forecast_as_it_was_issued(dirs):
    data_dir, _ = dirs
    log_forecast(data_dir, "2026-11-15", "2026-11-14 15:17:42", level=9e6)
    log_forecast(data_dir, "2026-11-16", "2026-11-15 15:20:05", level=1e7)

    response = client_of(dirs).get("/forecast")

    assert response.status_code == 200
    body = response.json()
    hours = body.pop("hours")
    assert body == {
        "target_day": "2026-11-16",
        "issued_at": "2026-11-15T15:20:05-05:00",
        "model_version": EARLIER,
        "demand_last_day": "2026-11-09",
        "generation_last_day": "2026-11-11",
        "timezone": BOGOTA,
        "unit": "kWh",
    }
    assert len(hours) == 24
    assert hours[0] == {"timestamp": "2026-11-16T00:00:00-05:00", "demand_kwh": 1e7}
    assert hours[23] == {"timestamp": "2026-11-16T23:00:00-05:00", "demand_kwh": 1e7 + 23000}


def test_forecast_is_unavailable_until_the_first_one_is_logged(dirs):
    response = client_of(dirs).get("/forecast")

    assert response.status_code == 503
    assert response.json() == {"detail": "no forecast has been issued yet"}


def test_health_says_which_model_and_days_the_service_holds(dirs):
    data_dir, _ = dirs
    log_forecast(data_dir, "2026-11-22", "2026-11-21 15:17", level=9e6)
    write_evaluated_days(data_dir, [2.0, 3.0])

    response = client_of(dirs).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_version": CURRENT,
        "last_forecast_day": "2026-11-22",
        "last_evaluated_day": "2026-11-16",
    }


def test_health_is_ok_before_anything_is_logged(dirs):
    assert client_of(dirs).get("/health").json() == {
        "status": "ok",
        "model_version": CURRENT,
        "last_forecast_day": None,
        "last_evaluated_day": None,
    }


def test_metrics_returns_the_backtest_and_the_live_error(dirs):
    data_dir, _ = dirs
    write_evaluated_days(data_dir, [9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    response = client_of(dirs).get("/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == CURRENT
    assert body["backtest"] == {
        "start": "2025-11-01",
        "end": "2026-10-31",
        "folds": 12,
        "models": [
            {"model": "seasonal-naive", "mae_kwh": 338628.444, "mape_percent": 3.4025},
            {"model": "moving-average", "mae_kwh": 300503.5935, "mape_percent": 3.034},
            {"model": "lightgbm", "mae_kwh": 194519.7729, "mape_percent": 1.9585},
        ],
    }
    assert body["live"] == {
        "evaluated_days": 8,
        "first_day": "2026-11-15",
        "last_day": "2026-11-22",
        "mae_kwh": pytest.approx(462500.0),
        "mape_percent": pytest.approx(4.625),
        "recent_days": 7,
        "recent_mape_percent": pytest.approx(4.0),
    }


def test_the_recent_error_waits_for_seven_evaluated_days(dirs):
    data_dir, _ = dirs
    write_evaluated_days(data_dir, [2.0, 3.0])

    live = client_of(dirs).get("/metrics").json()["live"]

    assert live["evaluated_days"] == 2
    assert live["mape_percent"] == pytest.approx(2.5)
    assert live["recent_mape_percent"] is None


def test_there_is_no_live_error_before_the_first_evaluation(dirs):
    assert client_of(dirs).get("/metrics").json()["live"] is None


def test_the_api_documents_its_three_endpoints(dirs):
    paths = client_of(dirs).get("/openapi.json").json()["paths"]
    assert set(paths) == {"/health", "/forecast", "/metrics"}


def test_the_bare_address_leads_to_the_documentation(dirs):
    client = client_of(dirs)

    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"
    assert client.get("/docs").status_code == 200


def test_the_service_does_not_start_without_a_model(tmp_path):
    with pytest.raises(RegistryError, match="run the `train` command first"):
        create_app(tmp_path / "data", tmp_path / "models")
