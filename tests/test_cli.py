from datetime import UTC, date, datetime

import pandas as pd
import pytest
from typer.testing import CliRunner

from demand_forecast import cli, config
from demand_forecast.data.ingest import ingest
from demand_forecast.data.xm_client import XMUnavailableError
from demand_forecast.models.registry import load_model, read_registry
from demand_forecast.models.train import LightGBMForecaster

runner = CliRunner()


class FakeXM:
    """Stand-in for `XMClient` that serves one synthetic table for every metric."""

    table: pd.DataFrame | None = None

    def fetch_hourly(self, metric_id: str, entity: str, start: date, end: date) -> pd.DataFrame:
        if self.table is None:
            raise XMUnavailableError("XM did not answer DemaCome/Sistema after 4 attempts")
        days = self.table["date"].dt.date
        return self.table[(days >= start) & (days <= end)].reset_index(drop=True)


class QuickModel(LightGBMForecaster):
    """The configured model with few trees, to keep the tests fast."""

    def __init__(self) -> None:
        super().__init__(rounds=10)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(
        config, "MLFLOW_TRACKING_URI", "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
    )
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.setattr(cli, "XMClient", FakeXM)
    monkeypatch.setattr(cli, "LightGBMForecaster", QuickModel)
    monkeypatch.setattr(FakeXM, "table", None)
    return data_dir


def tracked_runs() -> pd.DataFrame:
    import mlflow

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    return mlflow.search_runs(experiment_names=[config.MLFLOW_EXPERIMENT])


@pytest.fixture
def two_month_windows(monkeypatch):
    """Evaluate on 2026-09-01..2026-10-31 and select on 2026-07-01..2026-08-31: much faster."""
    monkeypatch.setattr(config, "BACKTEST_MONTHS", 2)


@pytest.fixture
def stored_history(data_dir, make_wide, monkeypatch):
    """Synthetic data from 2025-07-01 to 2026-11-06, already ingested."""
    monkeypatch.setattr(FakeXM, "table", make_wide("2025-07-01", 494))
    ingest(
        FakeXM(),
        today=date(2026, 11, 11),
        now=datetime(2026, 11, 12, 3, 0, tzinfo=UTC),
        data_dir=data_dir,
    )


def test_backtest_prints_every_model_and_the_breakdowns(stored_history):
    result = runner.invoke(cli.app, ["backtest", "--model", "seasonal-naive"])

    assert result.exit_code == 0, result.output
    assert "from 2025-11-01 to 2026-10-31: 12 monthly folds" in result.output
    table = result.output.split("seasonal-naive by day type")[0]
    assert "seasonal-naive" in table
    assert "moving-average" in table
    assert "lightgbm" in table
    assert "MAPE (%)" in table
    assert "sunday_or_holiday" in result.output
    assert "seasonal-naive by hour of the day" in result.output
    by_month = result.output.split("seasonal-naive by month")[1]
    assert "2025-11" in by_month
    assert "2026-10" in by_month
    worst = result.output.split("seasonal-naive: the 10 days with the largest error")[1]
    assert len(worst.strip().splitlines()) == 11


def test_backtest_can_break_down_the_model(stored_history, two_month_windows):
    result = runner.invoke(cli.app, ["backtest", "--model", "lightgbm"])
    assert result.exit_code == 0, result.output
    assert "from 2026-09-01 to 2026-10-31: 2 monthly folds" in result.output
    assert "lightgbm by day type" in result.output


def test_train_registers_the_model_with_its_backtest_and_tracks_the_run(
    stored_history, two_month_windows
):
    result = runner.invoke(cli.app, ["train"])

    assert result.exit_code == 0, result.output
    assert "Registered lightgbm-" in result.output
    model, entry = load_model(config.MODELS_DIR)
    assert model.booster is not None
    assert entry["data"]["demand_last_day"] == "2026-11-06"
    assert entry["data"]["last_target_day"] == "2026-11-06"
    assert entry["backtest"]["start"] == "2026-09-01"
    assert entry["backtest"]["end"] == "2026-10-31"
    assert entry["backtest"]["folds"] == 2
    assert set(entry["backtest"]["metrics"]) == {"seasonal-naive", "moving-average", "lightgbm"}
    assert set(entry["backtest"]["metrics"]["lightgbm"]) == {"mae", "mape"}

    (run,) = tracked_runs().to_dict(orient="records")
    assert run["tags.stage"] == "train"
    assert run["tags.model_version"] == entry["version"]
    assert run["tags.data_version"] == entry["data"]["fingerprint"]
    assert run["params.rounds"] == "10"
    registered_mape = entry["backtest"]["metrics"]["lightgbm"]["mape"]
    assert run["metrics.mape"] == pytest.approx(registered_mape, abs=1e-4)
    assert "metrics.mape_moving-average" in run


def test_train_without_data_registers_nothing(data_dir):
    result = runner.invoke(cli.app, ["train"])
    assert result.exit_code == 1
    assert "run the `ingest` command first" in result.output
    assert read_registry(config.MODELS_DIR) == {"current": None, "models": []}


def test_tune_scores_the_grid_on_the_months_before_the_evaluation_window(
    stored_history, two_month_windows, monkeypatch
):
    monkeypatch.setattr(config, "TUNING_GRID", {"rounds": [5], "num_leaves": [4, 8]})

    result = runner.invoke(cli.app, ["tune"])

    assert result.exit_code == 0, result.output
    assert "from 2026-07-01 to 2026-08-31: 2 monthly folds" in result.output
    assert "{'rounds': 5, 'num_leaves': 4}: MAPE" in result.output
    assert "{'rounds': 5, 'num_leaves': 8}: MAPE" in result.output
    runs = tracked_runs()
    assert len(runs) == 2
    assert set(runs["tags.stage"]) == {"tune"}
    assert set(runs["tags.window"]) == {"2026-07-01..2026-08-31"}
    assert set(runs["params.num_leaves"]) == {"4", "8"}


def test_tune_with_too_little_history_fails_cleanly(stored_history):
    result = runner.invoke(cli.app, ["tune"])
    assert result.exit_code == 1
    assert "error:" in result.output


def test_backtest_details_follow_the_chosen_model(stored_history, two_month_windows):
    result = runner.invoke(cli.app, ["backtest", "--model", "moving-average"])
    assert result.exit_code == 0, result.output
    assert "moving-average by day type" in result.output


def test_backtest_without_data_says_how_to_get_it(data_dir):
    result = runner.invoke(cli.app, ["backtest"])
    assert result.exit_code == 1
    assert "run the `ingest` command first" in result.output


def test_unknown_model_is_rejected(stored_history):
    result = runner.invoke(cli.app, ["backtest", "--model", "prophet"])
    assert result.exit_code == 2


def test_ingest_reports_what_changed(data_dir, make_wide, monkeypatch):
    monkeypatch.setattr(FakeXM, "table", make_wide("2026-01-01", 60))
    result = runner.invoke(cli.app, ["ingest"])

    assert result.exit_code == 0, result.output
    assert "last_complete_day" in result.output
    assert "2026-03-01" in result.output
    assert sorted(path.name for path in data_dir.iterdir()) == ["processed", "raw"]


def test_ingest_fails_with_a_clear_message_when_xm_is_down(data_dir):
    result = runner.invoke(cli.app, ["ingest"])
    assert result.exit_code == 1
    assert "error: XM did not answer" in result.output
    assert list(data_dir.iterdir()) == []


def test_backtest_with_too_little_history_fails_cleanly(data_dir, make_wide, monkeypatch):
    monkeypatch.setattr(FakeXM, "table", make_wide("2026-06-01", 150))
    ingest(
        FakeXM(),
        today=date(2026, 11, 11),
        now=datetime(2026, 11, 12, 3, 0, tzinfo=UTC),
        data_dir=data_dir,
    )
    result = runner.invoke(cli.app, ["backtest"])
    assert result.exit_code == 1
    assert "error: no published history to forecast 2025-10-01" in result.output
