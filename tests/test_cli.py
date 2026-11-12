from datetime import UTC, date, datetime

import pandas as pd
import pytest
from typer.testing import CliRunner

from demand_forecast import cli, config
from demand_forecast.data.ingest import ingest
from demand_forecast.data.xm_client import XMUnavailableError

runner = CliRunner()


class FakeXM:
    """Stand-in for `XMClient` that serves one synthetic table for every metric."""

    table: pd.DataFrame | None = None

    def fetch_hourly(self, metric_id: str, entity: str, start: date, end: date) -> pd.DataFrame:
        if self.table is None:
            raise XMUnavailableError("XM did not answer DemaCome/Sistema after 4 attempts")
        days = self.table["date"].dt.date
        return self.table[(days >= start) & (days <= end)].reset_index(drop=True)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cli, "XMClient", FakeXM)
    monkeypatch.setattr(FakeXM, "table", None)
    return tmp_path


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


def test_backtest_prints_both_baselines_and_the_breakdowns(stored_history):
    result = runner.invoke(cli.app, ["backtest", "--model", "seasonal-naive"])

    assert result.exit_code == 0, result.output
    assert "from 2025-11-01 to 2026-10-31: 12 monthly folds" in result.output
    table = result.output.split("seasonal-naive by day type")[0]
    assert "seasonal-naive" in table
    assert "moving-average" in table
    assert "MAPE (%)" in table
    assert "sunday_or_holiday" in result.output
    assert "seasonal-naive by hour of the day" in result.output


def test_backtest_details_follow_the_chosen_model(stored_history):
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
