"""HTTP API: the forecast in force, the model behind it and its error."""

from collections.abc import Callable
from importlib import metadata
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from demand_forecast import config
from demand_forecast.data import store
from demand_forecast.models.backtest import History
from demand_forecast.models.registry import load_model
from demand_forecast.serving.forecast import bogota_now, issue_forecast
from demand_forecast.serving.schemas import (
    BacktestMetrics,
    ForecastResponse,
    HealthResponse,
    HourlyForecast,
    MetricsResponse,
    ModelScore,
)


def create_app(
    data_dir: Path = config.DATA_DIR,
    models_dir: Path = config.MODELS_DIR,
    *,
    clock: Callable[[], pd.Timestamp] = bogota_now,
) -> FastAPI:
    """Return the API with the current model and the published history loaded.

    Raises when there is no registered model or no processed data: a service that cannot
    forecast must not start. `clock` returns the timezone-aware current time.
    """
    model, entry = load_model(models_dir)
    version = entry["version"]
    history = History(
        demand=store.load_series(config.DEMAND.name, data_dir),
        generation=store.load_series(config.GENERATION.name, data_dir),
    )
    health = HealthResponse(
        status="ok",
        model_version=version,
        demand_last_day=history.demand.index[-1].date(),
        generation_last_day=history.generation.index[-1].date(),
    )
    backtest = entry["backtest"]
    metrics = MetricsResponse(
        model_version=version,
        backtest=BacktestMetrics(
            start=backtest["start"],
            end=backtest["end"],
            folds=backtest["folds"],
            models=[
                ModelScore(model=name, mae_kwh=score["mae"], mape_percent=score["mape"])
                for name, score in backtest["metrics"].items()
            ],
        ),
    )

    app = FastAPI(
        title="Colombia day-ahead electricity demand forecast",
        description=(
            f"Hourly demand of the national grid ({config.DEMAND.metric_id}, XM) forecast one "
            "day ahead."
        ),
        version=metadata.version("demand-forecast"),
    )

    @app.get("/", include_in_schema=False)
    def get_root() -> RedirectResponse:
        """Send a visitor of the bare address to the interactive documentation."""
        return RedirectResponse("/docs")

    @app.get("/health")
    def get_health() -> HealthResponse:
        """Tell that the service is up, with the model and the data it holds."""
        return health

    @app.get("/forecast")
    def get_forecast() -> ForecastResponse:
        """Return the forecast for tomorrow, made with the data the service holds."""
        try:
            forecast = issue_forecast(model, version, history, now=clock())
        except ValueError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return ForecastResponse(
            target_day=forecast.target_day.date(),
            issued_at=forecast.issued_at.to_pydatetime(),
            model_version=forecast.model_version,
            demand_last_day=forecast.demand_last_day.date(),
            generation_last_day=forecast.generation_last_day.date(),
            timezone=config.TIMEZONE,
            unit="kWh",
            hours=[
                HourlyForecast(timestamp=timestamp.to_pydatetime(), demand_kwh=value)
                for timestamp, value in forecast.values.items()
            ],
        )

    @app.get("/metrics")
    def get_metrics() -> MetricsResponse:
        """Return the walk-forward backtest that backs the current model."""
        return metrics

    return app
