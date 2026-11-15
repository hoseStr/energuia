"""The daily job: ingest what XM published, score past forecasts and log tomorrow's."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from demand_forecast import config
from demand_forecast.data import store
from demand_forecast.data.ingest import HourlySource, SeriesReport, ingest
from demand_forecast.models.backtest import History
from demand_forecast.models.registry import load_model
from demand_forecast.monitoring.evaluate import evaluate, recent_mape, write_metrics
from demand_forecast.serving.forecast import Forecast, issue_forecast
from demand_forecast.serving.log import append_forecast, latest_forecast, read_log


class StaleDataError(Exception):
    """What XM has published is too old to back a forecast. No forecast was logged."""


@dataclass(frozen=True)
class DailyReport:
    """What one run of the daily job did."""

    ingest: list[SeriesReport]
    evaluated_days: int
    recent_mape: float | None
    forecast: Forecast
    issued: bool


def _check_fresh(history: History, issue_day: pd.Timestamp, max_extra_days: int) -> None:
    usual = (
        ("demand", history.demand, config.DATA_LAG_DAYS),
        ("generation", history.generation, config.GENERATION_LAG_DAYS),
    )
    for name, series, lag_days in usual:
        last_day = series.index[-1].normalize()
        oldest_allowed = issue_day - pd.Timedelta(days=lag_days + max_extra_days)
        if last_day < oldest_allowed:
            raise StaleDataError(
                f"the {name} is published up to {last_day.date()}, but a forecast issued on "
                f"{issue_day.date()} needs it up to {oldest_allowed.date()} at least "
                f"({lag_days} days of usual lag plus {max_extra_days} of tolerance)"
            )


def run_daily(
    source: HourlySource,
    *,
    now: pd.Timestamp,
    data_dir: Path = config.DATA_DIR,
    models_dir: Path = config.MODELS_DIR,
    max_extra_days: int = config.MAX_EXTRA_LAG_DAYS,
) -> DailyReport:
    """Run the daily job once, as of `now` (timezone-aware).

    The data is ingested and the error of the logged forecasts is rebuilt first. Then the
    forecast for tomorrow is logged, unless the log already has it (a second run on the same
    day changes nothing). Raises without logging a forecast when XM does not answer, when the
    published data is too old, or when there is no registered model.
    """
    local_now = now.tz_convert(config.TIMEZONE)
    ingested = ingest(source, today=local_now.date(), now=local_now, data_dir=data_dir)
    history = History(
        demand=store.load_series(config.DEMAND.name, data_dir),
        generation=store.load_series(config.GENERATION.name, data_dir),
    )
    log = read_log(data_dir)
    daily = evaluate(log, history.demand)
    write_metrics(daily, data_dir)

    def report(forecast: Forecast, issued: bool) -> DailyReport:
        return DailyReport(
            ingest=ingested,
            evaluated_days=len(daily),
            recent_mape=recent_mape(daily),
            forecast=forecast,
            issued=issued,
        )

    issue_day = local_now.normalize()
    target_day = issue_day + pd.Timedelta(days=config.HORIZON_DAYS)
    logged = latest_forecast(log)
    if logged is not None and logged.target_day == target_day:
        return report(logged, issued=False)

    _check_fresh(history, issue_day, max_extra_days)
    model, entry = load_model(models_dir)
    forecast = issue_forecast(model, entry["version"], history, now=local_now)
    append_forecast(forecast, data_dir)
    return report(forecast, issued=True)
