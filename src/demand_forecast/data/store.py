"""Parquet storage, one file per year so that a daily update only rewrites the current year."""

from pathlib import Path

import pandas as pd

from demand_forecast import config


def raw_dir(series: str, data_dir: Path = config.DATA_DIR) -> Path:
    """Folder of the raw (wide, as published) table of a series."""
    return data_dir / "raw" / series


def processed_dir(series: str, data_dir: Path = config.DATA_DIR) -> Path:
    """Folder of the processed (hourly, complete days only) table of a series."""
    return data_dir / "processed" / series


def read_table(directory: Path) -> pd.DataFrame | None:
    """Return every yearly file of `directory` as one table, or None when there is none."""
    files = sorted(directory.glob("*.parquet"))
    if not files:
        return None
    return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)


def write_table(table: pd.DataFrame, directory: Path, time_column: str) -> list[Path]:
    """Store `table` split by the year of `time_column`; return the files that changed.

    Files whose content is already identical are left untouched, and files of years that are
    no longer in the table are removed.
    """
    directory.mkdir(parents=True, exist_ok=True)
    changed = []
    wanted = set()
    for year, part in table.groupby(table[time_column].dt.year):
        path = directory / f"{year}.parquet"
        wanted.add(path)
        part = part.reset_index(drop=True)
        if path.exists() and pd.read_parquet(path).equals(part):
            continue
        temporary = path.with_suffix(".parquet.tmp")
        part.to_parquet(temporary, index=False)
        temporary.replace(path)
        changed.append(path)
    for stale in set(directory.glob("*.parquet")) - wanted:
        stale.unlink()
        changed.append(stale)
    return sorted(changed)


def load_series(series: str, data_dir: Path = config.DATA_DIR) -> pd.Series:
    """Return the processed hourly values of a series, indexed by timestamp."""
    table = read_table(processed_dir(series, data_dir))
    if table is None:
        raise FileNotFoundError(
            f"no processed data for '{series}' in {data_dir}; run the `ingest` command first"
        )
    return table.set_index("timestamp")["value"].rename(series)
