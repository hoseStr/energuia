import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from demand_forecast import config
from demand_forecast.data import store
from demand_forecast.data.ingest import IngestError, ingest, merge_raw
from demand_forecast.data.schema import HOUR_COLUMNS
from demand_forecast.data.xm_client import XMUnavailableError, empty_wide

TODAY = date(2026, 11, 10)
NOW = datetime(2026, 11, 11, 3, 46, tzinfo=UTC)
LATER = NOW + timedelta(days=1)
BOGOTA = "America/Bogota"


class FakeSource:
    """Serve slices of in-memory wide tables, keyed by XM metric id."""

    def __init__(self, tables: dict[str, pd.DataFrame | Exception]) -> None:
        self.tables = tables
        self.calls: list[tuple[str, date, date]] = []

    def fetch_hourly(self, metric_id: str, entity: str, start: date, end: date) -> pd.DataFrame:
        self.calls.append((metric_id, start, end))
        table = self.tables[metric_id]
        if isinstance(table, Exception):
            raise table
        days = table["date"].dt.date
        return table[(days >= start) & (days <= end)].reset_index(drop=True)


@pytest.fixture
def source(demand_wide, generation_wide) -> FakeSource:
    return FakeSource({"DemaCome": demand_wide, "Gene": generation_wide})


def tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        f"{layer}/{spec.name}": store.read_table(data_dir / layer / spec.name)
        for layer in ("raw", "processed")
        for spec in config.SERIES
    }


def files(data_dir: Path) -> dict[str, int]:
    return {str(path): path.stat().st_mtime_ns for path in data_dir.rglob("*.parquet")}


def complete_the_partial_day(source: FakeSource, generation_wide: pd.DataFrame) -> None:
    """Replace the partial 2026-11-06 demand with (almost exactly) that day's generation."""
    row = source.tables["DemaCome"]["date"] == "2026-11-06"
    filled = generation_wide.loc[generation_wide["date"] == "2026-11-06", HOUR_COLUMNS]
    source.tables["DemaCome"].loc[row, HOUR_COLUMNS] = filled.to_numpy() * 1.0001


class TestFirstRun:
    def test_requests_the_whole_history(self, tmp_path, source):
        ingest(source, today=TODAY, now=NOW, data_dir=tmp_path)
        assert source.calls == [
            ("DemaCome", config.XM_HISTORY_START, TODAY),
            ("Gene", config.XM_HISTORY_START, TODAY),
        ]

    def test_raw_keeps_every_published_day_including_partial_ones(self, tmp_path, source):
        ingest(source, today=TODAY, now=NOW, data_dir=tmp_path)
        raw = store.read_table(store.raw_dir("demand", tmp_path))
        assert len(raw) == 38
        assert raw["date"].max() == pd.Timestamp("2026-11-07")
        assert (raw["fetched_at"] == pd.Timestamp(NOW)).all()

    def test_processed_only_has_complete_days(self, tmp_path, source):
        reports = {r.series: r for r in ingest(source, today=TODAY, now=NOW, data_dir=tmp_path)}

        demand = store.load_series("demand", tmp_path)
        generation = store.load_series("generation", tmp_path)
        assert demand.index.max() == pd.Timestamp("2026-11-05 23:00", tz=BOGOTA)
        assert generation.index.max() == pd.Timestamp("2026-11-07 23:00", tz=BOGOTA)
        assert len(demand) == 36 * 24
        assert reports["demand"].last_published_day == date(2026, 11, 7)
        assert reports["demand"].last_complete_day == date(2026, 11, 5)
        assert reports["demand"].days_added == 38
        assert reports["generation"].last_complete_day == date(2026, 11, 7)


class TestIdempotence:
    def test_a_second_run_changes_nothing(self, tmp_path, source):
        ingest(source, today=TODAY, now=NOW, data_dir=tmp_path)
        tables_before, files_before = tables(tmp_path), files(tmp_path)

        reports = ingest(source, today=TODAY, now=LATER, data_dir=tmp_path)

        assert files(tmp_path) == files_before
        for name, table in tables(tmp_path).items():
            pd.testing.assert_frame_equal(table, tables_before[name])
        for report in reports:
            assert (report.days_added, report.days_revised, report.files_changed) == (0, 0, 0)

    def test_a_full_run_over_existing_data_changes_nothing_either(self, tmp_path, source):
        ingest(source, today=TODAY, now=NOW, data_dir=tmp_path)
        files_before = files(tmp_path)

        ingest(source, today=TODAY, now=LATER, data_dir=tmp_path, full=True)

        assert source.calls[-1] == ("Gene", config.XM_HISTORY_START, TODAY)
        assert files(tmp_path) == files_before


class TestIncrementalRun:
    def test_only_the_recent_window_is_requested(self, tmp_path, source):
        ingest(source, today=TODAY, now=NOW, data_dir=tmp_path)
        ingest(source, today=TODAY, now=LATER, data_dir=tmp_path)
        window_start = TODAY - timedelta(days=config.REDOWNLOAD_DAYS - 1)
        assert source.calls[2:] == [
            ("DemaCome", window_start, TODAY),
            ("Gene", window_start, TODAY),
        ]

    def test_a_long_pause_is_caught_up_from_the_last_stored_day(self, tmp_path, source):
        ingest(source, today=TODAY, now=NOW, data_dir=tmp_path)
        ingest(source, today=date(2027, 3, 1), now=LATER, data_dir=tmp_path)
        assert source.calls[2] == ("DemaCome", date(2026, 11, 7), date(2027, 3, 1))

    def test_a_partial_day_enters_processed_once_xm_completes_it(
        self, tmp_path, source, generation_wide
    ):
        ingest(source, today=TODAY, now=NOW, data_dir=tmp_path)
        complete_the_partial_day(source, generation_wide)

        reports = {r.series: r for r in ingest(source, today=TODAY, now=LATER, data_dir=tmp_path)}

        assert (reports["demand"].days_added, reports["demand"].days_revised) == (0, 1)
        assert reports["demand"].last_complete_day == date(2026, 11, 6)
        demand = store.load_series("demand", tmp_path)
        assert demand.index.max() == pd.Timestamp("2026-11-06 23:00", tz=BOGOTA)
        raw = store.read_table(store.raw_dir("demand", tmp_path)).set_index("date")
        assert raw.loc["2026-11-06", "fetched_at"] == pd.Timestamp(LATER)
        assert (raw.drop(pd.Timestamp("2026-11-06"))["fetched_at"] == pd.Timestamp(NOW)).all()

    def test_a_revised_complete_day_is_overwritten(self, tmp_path, source):
        ingest(source, today=TODAY, now=NOW, data_dir=tmp_path)
        row = source.tables["DemaCome"]["date"] == "2026-10-20"
        source.tables["DemaCome"].loc[row, "hour_10"] *= 1.0004

        reports = ingest(source, today=TODAY, now=LATER, data_dir=tmp_path)

        assert reports[0].days_revised == 1
        demand = store.load_series("demand", tmp_path)
        revised = source.tables["DemaCome"].loc[row, "hour_10"].iloc[0]
        assert demand.loc[pd.Timestamp("2026-10-20 09:00", tz=BOGOTA)] == revised

    def test_new_days_are_appended(self, tmp_path, source, demand_wide, generation_wide):
        source.tables = {"DemaCome": demand_wide.iloc[:-5], "Gene": generation_wide.iloc[:-5]}
        ingest(source, today=date(2026, 11, 5), now=NOW, data_dir=tmp_path)
        source.tables = {"DemaCome": demand_wide, "Gene": generation_wide}

        reports = ingest(source, today=TODAY, now=LATER, data_dir=tmp_path)

        assert reports[0].days_added == 5
        assert len(store.read_table(store.raw_dir("demand", tmp_path))) == 38


class TestQualityGates:
    def test_demand_waits_for_generation_to_confirm_it(self, tmp_path, source, generation_wide):
        complete_the_partial_day(source, generation_wide)
        source.tables["Gene"] = generation_wide.iloc[:-2]

        reports = {r.series: r for r in ingest(source, today=TODAY, now=NOW, data_dir=tmp_path)}

        assert reports["generation"].last_complete_day == date(2026, 11, 5)
        assert reports["demand"].last_complete_day == date(2026, 11, 5)

    def test_complete_days_after_a_hole_are_left_out_with_a_warning(self, tmp_path, source, caplog):
        row = source.tables["DemaCome"]["date"] == "2026-11-03"
        source.tables["DemaCome"].loc[row, HOUR_COLUMNS] *= 0.3

        with caplog.at_level(logging.WARNING):
            reports = ingest(source, today=TODAY, now=NOW, data_dir=tmp_path)

        assert reports[0].last_complete_day == date(2026, 11, 2)
        assert "2 complete day(s) after 2026-11-02" in caplog.text


class TestFailures:
    def test_no_data_from_xm_is_an_error_and_writes_nothing(self, tmp_path, source):
        source.tables["Gene"] = empty_wide()
        with pytest.raises(IngestError, match="no Gene data"):
            ingest(source, today=TODAY, now=NOW, data_dir=tmp_path)
        assert files(tmp_path) == {}

    def test_an_outage_midway_leaves_existing_data_untouched(self, tmp_path, source):
        ingest(source, today=TODAY, now=NOW, data_dir=tmp_path)
        files_before = files(tmp_path)
        source.tables["DemaCome"].loc[0, "hour_01"] += 1
        source.tables["Gene"] = XMUnavailableError("XM did not answer")

        with pytest.raises(XMUnavailableError):
            ingest(source, today=TODAY, now=LATER, data_dir=tmp_path)

        assert files(tmp_path) == files_before

    def test_an_incomplete_first_day_is_an_error(self, tmp_path, source):
        source.tables["DemaCome"].loc[0, "hour_01"] = float("nan")
        with pytest.raises(IngestError, match="first day of 'demand'"):
            ingest(source, today=TODAY, now=NOW, data_dir=tmp_path)
        assert files(tmp_path) == {}

    def test_implausible_values_fail_validation_and_write_nothing(self, tmp_path, source):
        # As if XM started publishing MWh instead of kWh.
        for table in source.tables.values():
            table[HOUR_COLUMNS] /= 1000
        with pytest.raises(IngestError, match="processed demand table failed validation"):
            ingest(source, today=TODAY, now=NOW, data_dir=tmp_path)
        assert files(tmp_path) == {}


class TestMergeRaw:
    AT = pd.Timestamp("2026-11-11 03:46", tz="UTC")

    def test_first_download_is_stored_as_is(self, demand_wide):
        merged, added, revised = merge_raw(None, demand_wide, self.AT)
        assert (added, revised) == (38, 0)
        assert list(merged.columns) == ["date", *HOUR_COLUMNS, "fetched_at"]

    def test_days_missing_from_the_new_download_are_kept(self, demand_wide):
        stored, _, _ = merge_raw(None, demand_wide, self.AT)
        merged, added, revised = merge_raw(stored, demand_wide.iloc[30:], self.AT)
        assert (added, revised) == (0, 0)
        assert len(merged) == 38

    def test_two_nan_hours_count_as_unchanged(self, demand_wide):
        demand_wide.loc[37, "hour_24"] = float("nan")
        stored, _, _ = merge_raw(None, demand_wide, self.AT)
        _, added, revised = merge_raw(stored, demand_wide, self.AT + pd.Timedelta(days=1))
        assert (added, revised) == (0, 0)
