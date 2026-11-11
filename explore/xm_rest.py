"""Minimal direct REST helper for Phase 0 exploration (no pydataxm)."""

import datetime as dt

import pandas as pd
import requests

BASE = "https://servapibi.xm.com.co"


def fetch(
    metric: str, entity: str, start: dt.date, end: dt.date, period: str = "hourly"
) -> pd.DataFrame:
    """Fetch one window (<= 30 days) and return a wide DataFrame with a Date column."""
    body = {
        "MetricId": metric,
        "StartDate": str(start),
        "EndDate": str(end),
        "Entity": entity,
        "Filter": [],
    }
    r = requests.post(f"{BASE}/{period}", json=body, timeout=120)
    if r.status_code == 400 and "No existen datos" in r.text:
        return pd.DataFrame()
    r.raise_for_status()
    items = r.json().get("Items", [])
    key = {"hourly": "HourlyEntities", "daily": "DailyEntities"}[period]
    if not items:
        return pd.DataFrame()
    df = pd.json_normalize(items, key, "Date", sep="_")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def hour_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("Values_Hour")]


def daily_total(df: pd.DataFrame) -> pd.Series:
    """Sum hourly columns per Date (summing across entities if several)."""
    vals = df[hour_cols(df)].apply(pd.to_numeric, errors="coerce")
    return vals.sum(axis=1).groupby(df["Date"]).sum()
