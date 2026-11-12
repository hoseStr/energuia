"""Command line entry point: `python -m demand_forecast.cli <command>`."""

import logging
from dataclasses import asdict
from enum import StrEnum
from typing import Annotated

import pandas as pd
import typer

from demand_forecast import config
from demand_forecast.data import store
from demand_forecast.data.ingest import IngestError, ingest
from demand_forecast.data.xm_client import XMClient, XMError
from demand_forecast.models import metrics
from demand_forecast.models.backtest import evaluation_window, make_folds, run_backtest
from demand_forecast.models.baselines import BASELINES


class ModelName(StrEnum):
    """Models the `backtest` command can evaluate."""

    SEASONAL_NAIVE = "seasonal-naive"
    MOVING_AVERAGE = "moving-average"


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Day-ahead hourly electricity demand forecast for Colombia.",
)


@app.callback()
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@app.command("ingest")
def ingest_command(
    full: Annotated[
        bool, typer.Option("--full", help="Request the whole history instead of the last days.")
    ] = False,
) -> None:
    """Download what XM published and rebuild the processed tables."""
    now = pd.Timestamp.now(tz=config.TIMEZONE)
    try:
        reports = ingest(XMClient(), today=now.date(), now=now, data_dir=config.DATA_DIR, full=full)
    except (XMError, IngestError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(pd.DataFrame([asdict(report) for report in reports]).to_string(index=False))


@app.command("backtest")
def backtest_command(
    model: Annotated[
        ModelName, typer.Option("--model", help="Model whose error is broken down in detail.")
    ] = ModelName.SEASONAL_NAIVE,
) -> None:
    """Evaluate a model against both baselines on the last 12 whole months of data."""
    try:
        demand = store.load_series(config.DEMAND.name, config.DATA_DIR)
        start, end = evaluation_window(demand.index.max().normalize())
        folds = make_folds(start, end)
        results = {name: run_backtest(demand, build(), folds) for name, build in BASELINES.items()}
    except (FileNotFoundError, ValueError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(
        f"Backtest of {config.DEMAND.metric_id} from {start.date()} to {end.date()}: "
        f"{len(folds)} monthly folds, forecasts issued {config.HORIZON_DAYS} day ahead "
        f"with data up to {config.DATA_LAG_DAYS} days before the issue day.\n"
    )
    summary = pd.DataFrame({name: metrics.overall(table) for name, table in results.items()}).T
    typer.echo(_format(summary.rename_axis("model")))
    typer.echo(f"\n{model.value} by day type")
    typer.echo(_format(metrics.by_day_type(results[model.value])))
    typer.echo(f"\n{model.value} by hour of the day")
    typer.echo(_format(metrics.by_hour(results[model.value])))


def _format(table: pd.DataFrame) -> str:
    """Render an error table with the MAE in MWh and the MAPE in percent."""
    shown = pd.DataFrame(
        {"MAE (MWh)": (table["mae"] / 1e3).round(1), "MAPE (%)": table["mape"].round(2)}
    )
    return shown.to_string()


if __name__ == "__main__":
    app()
