import pandas as pd
import pytest

from demand_forecast.data import store
from demand_forecast.data.transform import hourly_table


@pytest.fixture
def table(make_wide) -> pd.DataFrame:
    """Hourly table that spans two years: 2025-12-02..2026-01-30."""
    return hourly_table(make_wide("2025-12-02", 60))


def test_round_trip_keeps_values_and_dtypes(tmp_path, table):
    store.write_table(table, tmp_path, "timestamp")
    pd.testing.assert_frame_equal(store.read_table(tmp_path), table)


def test_one_file_per_local_year(tmp_path, table):
    changed = store.write_table(table, tmp_path, "timestamp")
    assert [path.name for path in changed] == ["2025.parquet", "2026.parquet"]
    year_2025 = pd.read_parquet(tmp_path / "2025.parquet")
    assert year_2025["timestamp"].max() == pd.Timestamp("2025-12-31 23:00", tz="America/Bogota")


def test_writing_the_same_table_again_touches_nothing(tmp_path, table):
    store.write_table(table, tmp_path, "timestamp")
    before = {path.name: path.stat().st_mtime_ns for path in tmp_path.iterdir()}

    assert store.write_table(table, tmp_path, "timestamp") == []
    assert {path.name: path.stat().st_mtime_ns for path in tmp_path.iterdir()} == before


def test_only_the_years_that_changed_are_rewritten(tmp_path, table):
    store.write_table(table, tmp_path, "timestamp")
    table.loc[len(table) - 1, "value"] += 1

    changed = store.write_table(table, tmp_path, "timestamp")

    assert [path.name for path in changed] == ["2026.parquet"]
    pd.testing.assert_frame_equal(store.read_table(tmp_path), table)


def test_years_no_longer_in_the_table_are_removed(tmp_path, table):
    store.write_table(table, tmp_path, "timestamp")
    only_2025 = table[table["timestamp"].dt.year == 2025]

    changed = store.write_table(only_2025, tmp_path, "timestamp")

    assert [path.name for path in changed] == ["2026.parquet"]
    assert [path.name for path in tmp_path.iterdir()] == ["2025.parquet"]


def test_read_table_without_files_returns_none(tmp_path):
    assert store.read_table(tmp_path / "missing") is None


def test_load_series_returns_values_indexed_by_timestamp(tmp_path, table):
    store.write_table(table, store.processed_dir("demand", tmp_path), "timestamp")
    series = store.load_series("demand", tmp_path)
    assert series.name == "demand"
    assert series.index.equals(pd.DatetimeIndex(table["timestamp"]))
    assert list(series) == list(table["value"])


def test_load_series_explains_what_to_do_when_there_is_no_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="ingest"):
        store.load_series("demand", tmp_path)
