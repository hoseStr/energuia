"""Keep the raw and processed tables in step with what XM publishes."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Protocol

import pandas as pd
import pandera.errors

from demand_forecast import config
from demand_forecast.data import completeness, schema, store, transform

logger = logging.getLogger(__name__)


class IngestError(Exception):
    """The ingestion cannot produce trustworthy tables. Nothing was written."""


class HourlySource(Protocol):
    """Anything that downloads hourly series the way `XMClient` does."""

    def fetch_hourly(self, metric_id: str, entity: str, start: date, end: date) -> pd.DataFrame: ...


@dataclass(frozen=True)
class SeriesReport:
    """What one ingestion run did to one series."""

    series: str
    requested_from: date
    days_added: int
    days_revised: int
    last_published_day: date
    last_complete_day: date
    files_changed: int


def _validated(
    validate: Callable[[pd.DataFrame], pd.DataFrame], table: pd.DataFrame, name: str
) -> pd.DataFrame:
    try:
        return validate(table)
    except pandera.errors.SchemaErrors as error:
        cases = error.failure_cases[["column", "check", "failure_case"]].head(10)
        raise IngestError(
            f"the {name} table failed validation:\n{cases.to_string(index=False)}"
        ) from error


def merge_raw(
    stored: pd.DataFrame | None, fresh: pd.DataFrame, fetched_at: pd.Timestamp
) -> tuple[pd.DataFrame, int, int]:
    """Overwrite the stored days with the freshly downloaded ones.

    Return the merged table and how many days were added and how many revised. A day whose
    values did not change keeps its row untouched, so `fetched_at` tells when each day was
    last seen changing and running the ingestion twice writes nothing the second time.
    """
    fresh = fresh.assign(fetched_at=fetched_at).astype({"fetched_at": schema.FETCHED_AT_DTYPE})
    if stored is None:
        return fresh, len(fresh), 0
    old, new = stored.set_index("date"), fresh.set_index("date")
    common = new.index.intersection(old.index)
    before, after = old.loc[common, schema.HOUR_COLUMNS], new.loc[common, schema.HOUR_COLUMNS]
    unchanged = ((before == after) | (before.isna() & after.isna())).all(axis=1)
    new = new.drop(index=common[unchanged])
    if new.empty:
        return stored, 0, 0
    added = len(new.index.difference(old.index))
    merged = pd.concat([old.drop(index=new.index.intersection(old.index)), new])
    return merged.sort_index().reset_index(), added, len(new) - added


def build_processed(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Derive the hourly table of every series from the raw tables.

    Each hourly table ends at the last day of the unbroken run of complete days, so it never
    has gaps. Demand days must also be confirmed by that day's generation.
    """
    totals = {name: transform.daily_totals(wide) for name, wide in raw.items()}
    complete = {name: completeness.complete_days(total) for name, total in totals.items()}
    demand, generation = config.DEMAND.name, config.GENERATION.name
    complete[demand] &= completeness.confirmed_by_generation(
        totals[demand], totals[generation], complete[generation]
    )

    processed = {}
    for name, wide in raw.items():
        last = completeness.last_gapless_day(complete[name])
        if last is None:
            raise IngestError(f"the first day of '{name}' is not complete")
        stranded = int(complete[name][complete[name].index > last].sum())
        if stranded:
            logger.warning(
                "%s: %d complete day(s) after %s are left out because an earlier day is "
                "missing or incomplete",
                name,
                stranded,
                last.date(),
            )
        processed[name] = _validated(
            schema.validate_hourly,
            transform.hourly_table(wide[wide["date"] <= last]),
            f"processed {name}",
        )
    return processed


def ingest(
    source: HourlySource,
    *,
    today: date,
    now: datetime,
    data_dir: Path = config.DATA_DIR,
    full: bool = False,
) -> list[SeriesReport]:
    """Download what changed, then rebuild the processed tables.

    `today` is the current date in Bogota and `now` a timezone-aware instant. Without `full`,
    only the last `REDOWNLOAD_DAYS` days are requested (the whole history on the first run).
    Every series is downloaded and validated before anything is written.
    """
    fetched_at = pd.Timestamp(now).tz_convert("UTC").floor("s")
    raw, counts, requested = {}, {}, {}
    for spec in config.SERIES:
        stored = store.read_table(store.raw_dir(spec.name, data_dir))
        start = config.XM_HISTORY_START
        if stored is not None and not full:
            start = min(
                today - timedelta(days=config.REDOWNLOAD_DAYS - 1), stored["date"].max().date()
            )
        logger.info("downloading %s from %s to %s", spec.metric_id, start, today)
        fresh = source.fetch_hourly(spec.metric_id, spec.entity, start, today)
        if fresh.empty:
            raise IngestError(f"XM returned no {spec.metric_id} data for {start}..{today}")
        merged, added, revised = merge_raw(stored, fresh, fetched_at)
        raw[spec.name] = _validated(schema.validate_raw, merged, f"raw {spec.name}")
        counts[spec.name] = (added, revised)
        requested[spec.name] = start
    processed = build_processed(raw)

    reports = []
    for spec in config.SERIES:
        changed = store.write_table(raw[spec.name], store.raw_dir(spec.name, data_dir), "date")
        changed += store.write_table(
            processed[spec.name], store.processed_dir(spec.name, data_dir), "timestamp"
        )
        reports.append(
            SeriesReport(
                series=spec.name,
                requested_from=requested[spec.name],
                days_added=counts[spec.name][0],
                days_revised=counts[spec.name][1],
                last_published_day=raw[spec.name]["date"].max().date(),
                last_complete_day=processed[spec.name]["timestamp"].max().date(),
                files_changed=len(changed),
            )
        )
    return reports
