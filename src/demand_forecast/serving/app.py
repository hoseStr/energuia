"""HTTP API over what the daily job wrote: the latest forecast and its measured error."""

from importlib import metadata
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from demand_forecast import config
from demand_forecast.models.registry import registry_entry
from demand_forecast.monitoring.evaluate import read_metrics, recent_mape
from demand_forecast.serving.log import latest_forecast, read_log
from demand_forecast.serving.schemas import (
    BacktestMetrics,
    ForecastResponse,
    HealthResponse,
    HourlyForecast,
    LiveMetrics,
    MetricsResponse,
    ModelScore,
)


def create_app(data_dir: Path = config.DATA_DIR, models_dir: Path = config.MODELS_DIR) -> FastAPI:
    """Return the API over the forecast log, the stored errors and the model registry.

    The files are read once, here: the service shows what they held when it started, and it
    never computes a forecast of its own. Raises when there is no registered model.
    """
    entry = registry_entry(models_dir)
    version = entry["version"]
    forecast = latest_forecast(read_log(data_dir))
    daily = read_metrics(data_dir)

    health = HealthResponse(
        status="ok",
        model_version=version,
        last_forecast_day=None if forecast is None else forecast.target_day.date(),
        last_evaluated_day=None if daily.empty else daily["target_day"].iloc[-1].date(),
    )
    backtest = entry["backtest"]
    live = None
    if not daily.empty:
        live = LiveMetrics(
            evaluated_days=len(daily),
            first_day=daily["target_day"].iloc[0].date(),
            last_day=daily["target_day"].iloc[-1].date(),
            mae_kwh=daily["mae_kwh"].mean(),
            mape_percent=daily["mape_percent"].mean(),
            recent_days=config.ALERT_WINDOW_DAYS,
            recent_mape_percent=recent_mape(daily),
        )
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
        live=live,
    )

    app = FastAPI(
        title="Colombia day-ahead electricity demand forecast",
        description=(
            f"Hourly demand of the national grid ({config.DEMAND.metric_id}, XM) forecast one "
            "day ahead. Forecasts are issued and logged once a day; this service reads the log."
        ),
        version=metadata.version("demand-forecast"),
    )

    @app.get("/", include_in_schema=False)
    def get_root() -> RedirectResponse:
        """Send a visitor of the bare address to the interactive documentation."""
        return RedirectResponse("/docs")

    @app.get("/health")
    def get_health() -> HealthResponse:
        """Tell that the service is up, with the model and the days it holds."""
        return health

    @app.get("/forecast")
    def get_forecast() -> ForecastResponse:
        """Return the latest forecast in the log, exactly as it was issued."""
        if forecast is None:
            raise HTTPException(status_code=503, detail="no forecast has been issued yet")
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
        """Return the backtest of the current model and the error measured on logged days."""
        return metrics

    return app
