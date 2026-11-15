"""Response bodies of the API."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """The service is up, and this is what it holds."""

    status: Literal["ok"]
    model_version: str = Field(description="Model that will issue the next forecast.")
    last_forecast_day: date | None = Field(description="Last day with a logged forecast.")
    last_evaluated_day: date | None = Field(
        description="Last forecast day whose actual demand is already published."
    )


class HourlyForecast(BaseModel):
    """Forecast demand of one hour."""

    timestamp: datetime = Field(description="Start of the hour, in local time.")
    demand_kwh: float


class ForecastResponse(BaseModel):
    """The latest logged forecast, with the model and the data it was made from."""

    target_day: date
    issued_at: datetime = Field(description="When the daily job issued and logged it.")
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


class LiveMetrics(BaseModel):
    """Error of the logged forecasts against the demand published afterwards."""

    evaluated_days: int
    first_day: date
    last_day: date
    mae_kwh: float
    mape_percent: float
    recent_days: int = Field(description="Length, in evaluated days, of the recent window.")
    recent_mape_percent: float | None = Field(
        description="MAPE of the last `recent_days` evaluated days; null until there are enough."
    )


class MetricsResponse(BaseModel):
    """Error of the current model: in the backtest and, once days are evaluated, live."""

    model_version: str
    backtest: BacktestMetrics
    live: LiveMetrics | None = Field(description="Null until a logged forecast can be evaluated.")
