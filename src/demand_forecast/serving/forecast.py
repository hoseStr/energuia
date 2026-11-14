"""Issue the day-ahead forecast with a registered model and everything published so far."""

from dataclasses import dataclass

import pandas as pd

from demand_forecast import config
from demand_forecast.models.backtest import Forecaster, History


@dataclass(frozen=True)
class Forecast:
    """The 24 hourly values forecast for one day, and what they were made from."""

    issued_at: pd.Timestamp
    target_day: pd.Timestamp
    model_version: str
    demand_last_day: pd.Timestamp
    generation_last_day: pd.Timestamp
    values: pd.Series


def bogota_now() -> pd.Timestamp:
    """Return the current time in Bogota."""
    return pd.Timestamp.now(tz=config.TIMEZONE)


def issue_forecast(
    model: Forecaster,
    model_version: str,
    history: History,
    *,
    now: pd.Timestamp,
    horizon_days: int = config.HORIZON_DAYS,
) -> Forecast:
    """Forecast the day that falls `horizon_days` after the Bogota date of `now`.

    `history` is everything published so far, uncut: live, the stored tables only hold
    published days. It must end before the issue day; to replay a past issue day, cut the
    history with `history_at` first.
    """
    if now.tzinfo is None:
        raise ValueError("the issue time must be timezone-aware")
    issued_at = now.tz_convert(config.TIMEZONE)
    issue_day = issued_at.normalize()
    if history.generation is None or history.generation.empty or history.demand.empty:
        raise ValueError("no published demand and generation to issue a forecast")
    for name, series in (("demand", history.demand), ("generation", history.generation)):
        last_day = series.index[-1].normalize()
        if last_day >= issue_day:
            raise ValueError(
                f"the {name} history holds {last_day.date()}, but {issue_day.date()} was not "
                "over when the forecast was issued"
            )
    target_day = issue_day + pd.Timedelta(days=horizon_days)
    return Forecast(
        issued_at=issued_at,
        target_day=target_day,
        model_version=model_version,
        demand_last_day=history.demand.index[-1].normalize(),
        generation_last_day=history.generation.index[-1].normalize(),
        values=model.predict(history, target_day),
    )
