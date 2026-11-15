"""Command line entry point: `python -m demand_forecast.cli <command>`."""

import logging
from dataclasses import asdict
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
import uvicorn

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
from demand_forecast.models.promote import compare, decide
from demand_forecast.models.registry import (
    RegistryError,
    git_commit,
    load_model,
    record_decision,
    register_model,
    registry_entry,
)
from demand_forecast.models.train import LightGBMForecaster
from demand_forecast.models.tune import selection_window, tune
from demand_forecast.monitoring.drift import write_drift
from demand_forecast.monitoring.evaluate import alert_message, read_metrics, recent_mape
from demand_forecast.monitoring.report import write_panel
from demand_forecast.pipeline import StaleDataError, run_daily
from demand_forecast.serving.app import create_app
from demand_forecast.serving.forecast import bogota_now, issue_forecast
from demand_forecast.serving.log import LogError

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
        entry, summary, description = _train_and_register(demand, generation)
    except (FileNotFoundError, ValueError, RegistryError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    settings = {name: value for name, value in entry["model"].items() if name != "features"}
    tracking.log_run(
        "train",
        params={**settings.pop("params"), **settings},
        metrics={
            **summary.loc[LightGBMForecaster.name].to_dict(),
            **{f"mape_{name}": summary.loc[name, "mape"] for name in BASELINES},
        },
        tags={
            "stage": "train",
            "window": f"{entry['backtest']['start']}..{entry['backtest']['end']}",
            "data_version": entry["data"]["fingerprint"],
            "model_version": entry["version"],
            "commit": entry["commit"],
        },
    )
    typer.echo(f"Backtest of {config.DEMAND.metric_id} {description}\n")
    typer.echo(_format(summary))
    typer.echo("\n" + _registered(entry))


@app.command("retrain")
def retrain_command() -> None:
    """Register a retrained model only if retraining beat the current model since it was trained.

    The decision is written to the registry either way.
    """
    try:
        demand, generation = _load_series()
        champion, current = load_model(config.MODELS_DIR)
        trained_up_to = pd.Timestamp(current["data"]["last_target_day"], tz=config.TIMEZONE)
        comparison = compare(
            demand, generation, champion, trained_up_to, build=lambda: LightGBMForecaster()
        )
        promote, reason = decide(comparison)
        entry = _train_and_register(demand, generation)[0] if promote else None
        record_decision(
            {
                "decided_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "current_version": current["version"],
                "window_start": None if comparison.start is None else str(comparison.start.date()),
                "window_end": None if comparison.end is None else str(comparison.end.date()),
                "days": comparison.days,
                "current_mape": _rounded(comparison.champion_mape),
                "retrained_mape": _rounded(comparison.challenger_mape),
                "promoted": promote,
                "reason": reason,
                "new_version": None if entry is None else entry["version"],
            },
            config.MODELS_DIR,
        )
    except (FileNotFoundError, ValueError, RegistryError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    if comparison.days:
        typer.echo(
            f"{comparison.days} days published since model {current['version']} was trained "
            f"({comparison.start.date()} to {comparison.end.date()}): MAPE "
            f"{comparison.champion_mape:.2f} % for the current model and "
            f"{comparison.challenger_mape:.2f} % for models retrained "
            f"every {config.RETRAIN_REFIT_DAYS} days."
        )
    if entry is None:
        typer.echo(f"Kept model {current['version']}: {reason}.")
    else:
        typer.echo(f"Promoted a retrained model: {reason}.\n" + _registered(entry))


@app.command("forecast")
def forecast_command() -> None:
    """Forecast tomorrow with the current model and everything published so far."""
    try:
        demand, generation = _load_series()
        model, entry = load_model(config.MODELS_DIR)
        forecast = issue_forecast(
            model, entry["version"], History(demand=demand, generation=generation), now=bogota_now()
        )
    except (FileNotFoundError, ValueError, RegistryError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(
        f"Forecast of {config.DEMAND.metric_id} for {forecast.target_day.date()}, issued "
        f"{forecast.issued_at:%Y-%m-%d %H:%M} with model {forecast.model_version}: demand "
        f"published up to {forecast.demand_last_day.date()} and generation up to "
        f"{forecast.generation_last_day.date()}.\n"
    )
    table = pd.DataFrame({"MWh": (forecast.values / 1e3).round(1)})
    table.index = forecast.values.index.strftime("%H:%M")
    typer.echo(table.to_string())
    typer.echo(f"\nTotal: {forecast.values.sum() / 1e6:.1f} GWh")


@app.command("daily")
def daily_command() -> None:
    """Run the daily job: ingest, score the logged forecasts and log the forecast for tomorrow."""
    try:
        report = run_daily(
            XMClient(), now=bogota_now(), data_dir=config.DATA_DIR, models_dir=config.MODELS_DIR
        )
    except (
        XMError,
        IngestError,
        StaleDataError,
        RegistryError,
        LogError,
        FileNotFoundError,
        ValueError,
    ) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(pd.DataFrame([asdict(one) for one in report.ingest]).to_string(index=False))
    recent = ""
    if report.recent_mape is not None:
        recent = f"; MAPE of the last {config.ALERT_WINDOW_DAYS}: {report.recent_mape:.2f} %"
    typer.echo(f"\nEvaluated days: {report.evaluated_days}{recent}")
    forecast = report.forecast
    made = (
        f"with model {forecast.model_version}, demand published up to "
        f"{forecast.demand_last_day.date()} and generation up to "
        f"{forecast.generation_last_day.date()}"
    )
    if report.issued:
        typer.echo(f"Logged the forecast for {forecast.target_day.date()} {made}.")
    else:
        typer.echo(
            f"The forecast for {forecast.target_day.date()} was already logged at "
            f"{forecast.issued_at:%Y-%m-%d %H:%M} {made}."
        )


@app.command("alert")
def alert_command(
    output: Annotated[
        Path, typer.Option("--output", help="File to write the warning to, only if there is one.")
    ],
) -> None:
    """Compare the error of the last evaluated days with the backtest of the current model."""
    try:
        entry = registry_entry(config.MODELS_DIR)
    except RegistryError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    daily = read_metrics(config.DATA_DIR)
    backtest_mape = entry["backtest"]["metrics"][LightGBMForecaster.name]["mape"]
    message = alert_message(daily, backtest_mape)
    if message is not None:
        output.write_text(message + "\n", encoding="utf-8", newline="\n")
        typer.echo(message)
        return
    recent = recent_mape(daily)
    if recent is None:
        typer.echo(f"No alert: only {len(daily)} evaluated days so far.")
    else:
        typer.echo(
            f"No alert: the MAPE of the last {config.ALERT_WINDOW_DAYS} evaluated days is "
            f"{recent:.2f} % and the one of the backtest is {backtest_mape:.2f} %."
        )


@app.command("drift")
def drift_command(
    site_dir: Annotated[
        Path, typer.Option("--site-dir", help="Folder of the generated site.")
    ] = config.SITE_DIR,
) -> None:
    """Compare the features of the logged forecasts with the training period of the model."""
    try:
        summary = write_drift(config.DATA_DIR, config.MODELS_DIR, site_dir)
    except ImportError as error:
        typer.echo("error: Evidently is missing; run `uv sync --group monitoring`", err=True)
        raise typer.Exit(code=1) from error
    except (FileNotFoundError, ValueError, RegistryError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    if summary is None:
        typer.echo(f"No drift report: it needs {config.DRIFT_MIN_DAYS} logged forecast days.")
        return
    drifted = ", ".join(summary.drifted) or "none"
    typer.echo(
        f"Drift over {summary.current_days} logged days ({summary.current_start} to "
        f"{summary.current_end}) against {summary.reference_start} to {summary.reference_end}: "
        f"{len(summary.drifted)} of {len(summary.scores)} features drifted ({drifted})."
    )


@app.command("report")
def report_command(
    site_dir: Annotated[
        Path, typer.Option("--site-dir", help="Folder of the generated site.")
    ] = config.SITE_DIR,
) -> None:
    """Build the public panel from the forecast log, its error and the last drift report."""
    try:
        path = write_panel(config.DATA_DIR, config.MODELS_DIR, site_dir, now=bogota_now())
    except (FileNotFoundError, RegistryError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Wrote {path}")


@app.command("serve")
def serve_command(
    host: Annotated[str, typer.Option("--host", help="Interface to listen on.")] = config.API_HOST,
    port: Annotated[
        int, typer.Option("--port", envvar="PORT", help="Port to listen on.")
    ] = config.API_PORT,
) -> None:
    """Serve the forecast over HTTP with the current model and the stored data."""
    try:
        api = create_app(config.DATA_DIR, config.MODELS_DIR)
    except RegistryError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    uvicorn.run(api, host=host, port=port)


def _train_and_register(
    demand: pd.Series, generation: pd.Series
) -> tuple[dict[str, object], pd.DataFrame, str]:
    """Backtest the configured model, fit it on everything published and register it.

    Return the registry entry, the backtest summary and the description of the backtest.
    """
    start, end = evaluation_window(demand.index.max().normalize(), config.BACKTEST_MONTHS)
    folds = make_folds(start, end)
    summary = _summary(_evaluate(demand, generation, folds))
    model = LightGBMForecaster()
    model.fit(History(demand=demand, generation=generation))
    entry = register_model(
        model,
        trained_at=datetime.now(UTC),
        data={
            "demand_last_day": demand.index.max().date().isoformat(),
            "generation_last_day": generation.index.max().date().isoformat(),
            "fingerprint": store.fingerprint(demand, generation),
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
    return entry, summary, _describe(start, end, folds)


def _registered(entry: dict[str, object]) -> str:
    return (
        f"Registered {entry['file']} as the current model: trained on "
        f"{entry['data']['first_target_day']}..{entry['data']['last_target_day']} "
        f"({entry['data']['rows']} hours), data version {entry['data']['fingerprint']}."
    )


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


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
