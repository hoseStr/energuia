"""Command line entry point: `python -m demand_forecast.cli <command>`."""

import logging
from dataclasses import asdict
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated

import pandas as pd
import typer

from demand_forecast import config, tracking
from demand_forecast.data import store
from demand_forecast.data.ingest import IngestError, ingest
from demand_forecast.data.xm_client import XMClient, XMError
from demand_forecast.models import metrics
from demand_forecast.models.backtest import (
    Fold,
    History,
    evaluation_window,
    make_folds,
    run_backtest,
)
from demand_forecast.models.baselines import BASELINES
from demand_forecast.models.registry import RegistryError, git_commit, register_model
from demand_forecast.models.train import LightGBMForecaster
from demand_forecast.models.tune import selection_window, tune

WORST_DAYS = 10


class ModelName(StrEnum):
    """Models the `backtest` command can break down in detail."""

    SEASONAL_NAIVE = "seasonal-naive"
    MOVING_AVERAGE = "moving-average"
    LIGHTGBM = "lightgbm"


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
    """Evaluate the model and both baselines on the last 12 whole months of data."""
    try:
        demand, generation = _load_series()
        start, end = evaluation_window(demand.index.max().normalize(), config.BACKTEST_MONTHS)
        folds = make_folds(start, end)
        results = _evaluate(demand, generation, folds)
    except (FileNotFoundError, ValueError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Backtest of {config.DEMAND.metric_id} {_describe(start, end, folds)}\n")
    typer.echo(_format(_summary(results)))
    chosen = results[model.value]
    typer.echo(f"\n{model.value} by day type")
    typer.echo(_format(metrics.by_day_type(chosen)))
    typer.echo(f"\n{model.value} by hour of the day")
    typer.echo(_format(metrics.by_hour(chosen)))
    typer.echo(f"\n{model.value} by month")
    typer.echo(_format(metrics.by_month(chosen)))
    typer.echo(f"\n{model.value}: the {WORST_DAYS} days with the largest error")
    worst = metrics.by_day(chosen).nlargest(WORST_DAYS, "mape")
    worst.index = worst.index.strftime("%Y-%m-%d %a").rename(None)
    typer.echo(pd.concat([_shown(worst), worst["day_type"]], axis=1).to_string())


@app.command("tune")
def tune_command() -> None:
    """Score the hyperparameter grid on the 12 months before the evaluation window."""
    try:
        demand, generation = _load_series()
        evaluation_start, _ = evaluation_window(
            demand.index.max().normalize(), config.BACKTEST_MONTHS
        )
        start, end = selection_window(evaluation_start, config.BACKTEST_MONTHS)
        folds = make_folds(start, end)
        tags = {
            "stage": "tune",
            "window": f"{start.date()}..{end.date()}",
            "data_version": store.fingerprint(demand, generation),
        }

        def record(row: dict[str, object]) -> None:
            scores = {name: row[name] for name in ("mae", "mape")}
            candidate = {name: value for name, value in row.items() if name not in scores}
            tracking.log_run("tune", params=candidate, metrics=scores, tags=tags)
            typer.echo(f"{candidate}: MAPE {row['mape']:.3f} %")

        typer.echo(f"Tuning on {config.DEMAND.metric_id} {_describe(start, end, folds)}\n")
        table = tune(demand, generation, folds, config.TUNING_GRID, on_result=record)
    except (FileNotFoundError, ValueError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    scores = _shown(table)
    typer.echo("\n" + pd.concat([table.drop(columns=["mae", "mape"]), scores], axis=1).to_string())


@app.command("train")
def train_command() -> None:
    """Backtest the configured model, fit it on everything published and register it."""
    try:
        demand, generation = _load_series()
        start, end = evaluation_window(demand.index.max().normalize(), config.BACKTEST_MONTHS)
        folds = make_folds(start, end)
        summary = _summary(_evaluate(demand, generation, folds))
        model = LightGBMForecaster()
        model.fit(History(demand=demand, generation=generation))
        data_version = store.fingerprint(demand, generation)
        entry = register_model(
            model,
            trained_at=datetime.now(UTC),
            data={
                "demand_last_day": demand.index.max().date().isoformat(),
                "generation_last_day": generation.index.max().date().isoformat(),
                "fingerprint": data_version,
            },
            backtest={
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
                "folds": len(folds),
                "metrics": summary.round(4).to_dict(orient="index"),
            },
            commit=git_commit(),
            models_dir=config.MODELS_DIR,
        )
    except (FileNotFoundError, ValueError, RegistryError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    settings = {name: value for name, value in entry["model"].items() if name != "features"}
    tracking.log_run(
        "train",
        params={**settings.pop("params"), **settings},
        metrics={
            **summary.loc[model.name].to_dict(),
            **{f"mape_{name}": summary.loc[name, "mape"] for name in BASELINES},
        },
        tags={
            "stage": "train",
            "window": f"{start.date()}..{end.date()}",
            "data_version": data_version,
            "model_version": entry["version"],
            "commit": entry["commit"],
        },
    )
    typer.echo(f"Backtest of {config.DEMAND.metric_id} {_describe(start, end, folds)}\n")
    typer.echo(_format(summary))
    typer.echo(
        f"\nRegistered {entry['file']} as the current model: trained on "
        f"{entry['data']['first_target_day']}..{entry['data']['last_target_day']} "
        f"({entry['data']['rows']} hours), data version {data_version}."
    )


def _load_series() -> tuple[pd.Series, pd.Series]:
    """Return the processed demand and generation."""
    return (
        store.load_series(config.DEMAND.name, config.DATA_DIR),
        store.load_series(config.GENERATION.name, config.DATA_DIR),
    )


def _evaluate(
    demand: pd.Series, generation: pd.Series, folds: list[Fold]
) -> dict[str, pd.DataFrame]:
    """Backtest both baselines and the configured model on the same folds."""
    results = {name: run_backtest(demand, build(), folds) for name, build in BASELINES.items()}
    results[LightGBMForecaster.name] = run_backtest(
        demand, LightGBMForecaster(), folds, generation=generation
    )
    return results


def _summary(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the overall MAE and MAPE of every model, one row each."""
    table = pd.DataFrame({name: metrics.overall(table) for name, table in results.items()}).T
    return table.rename_axis("model")


def _describe(start: pd.Timestamp, end: pd.Timestamp, folds: list[Fold]) -> str:
    return (
        f"from {start.date()} to {end.date()}: {len(folds)} monthly folds, forecasts issued "
        f"{config.HORIZON_DAYS} day ahead with demand up to {config.DATA_LAG_DAYS} days and "
        f"generation up to {config.GENERATION_LAG_DAYS} days before the issue day."
    )


def _shown(table: pd.DataFrame) -> pd.DataFrame:
    """Return the error columns with the MAE in MWh and the MAPE in percent."""
    return pd.DataFrame(
        {"MAE (MWh)": (table["mae"] / 1e3).round(1), "MAPE (%)": table["mape"].round(2)}
    )


def _format(table: pd.DataFrame) -> str:
    return _shown(table).to_string()


if __name__ == "__main__":
    app()
