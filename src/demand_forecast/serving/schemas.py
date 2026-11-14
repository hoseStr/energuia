"""Response bodies of the API."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """The service is up, and this is what it holds."""

    status: Literal["ok"]
    model_version: str
    demand_last_day: date = Field(description="Last published day of demand the service holds.")
    generation_last_day: date = Field(
        description="Last published day of generation the service holds."
    )


class HourlyForecast(BaseModel):
    """Forecast demand of one hour."""

    timestamp: datetime = Field(description="Start of the hour, in local time.")
    demand_kwh: float


class ForecastResponse(BaseModel):
    """The day-ahead forecast, with the model and the data it was made from."""

    target_day: date
    issued_at: datetime = Field(description="When this forecast was computed.")
    model_version: str
    demand_last_day: date = Field(description="Last day of demand the forecast could see.")
    generation_last_day: date = Field(description="Last day of generation the forecast could see.")
    timezone: str
    unit: Literal["kWh"]
    hours: list[HourlyForecast]


class ModelScore(BaseModel):
    """Error of one model over the backtest window."""

    model: str
    mae_kwh: float
    mape_percent: float


class BacktestMetrics(BaseModel):
    """Walk-forward backtest: one fold per month, forecasts issued one day ahead."""

    start: date
    end: date
    folds: int
    models: list[ModelScore]


class MetricsResponse(BaseModel):
    """Error of the current model, next to the baselines on the same window."""

    model_version: str
    backtest: BacktestMetrics
