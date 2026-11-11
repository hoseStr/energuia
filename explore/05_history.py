"""Phase 0: find history start, download full history month by month, check format consistency."""

import datetime as dt
import sys
import time
from pathlib import Path

import pandas as pd
from xm_rest import fetch, hour_cols

OUT = Path(__file__).parent / "output"
pd.set_option("display.width", 250)

metric = sys.argv[1]
today = dt.date.today()

# Exact first date: probe Dec 1999 - Jan 2000
probe = fetch(metric, "Sistema", dt.date(2000, 1, 1), dt.date(2000, 1, 31))
print(metric, "first date available:", probe.Date.min().date() if not probe.empty else None)

frames, empty_months = [], []
months = pd.date_range("2000-01-01", today, freq="MS")
t0 = time.time()
for m0 in months:
    m1 = min((m0 + pd.offsets.MonthEnd(0)).date(), today)
    df = fetch(metric, "Sistema", m0.date(), m1)
    if df.empty:
        empty_months.append(str(m0.date()))
        continue
    df["_cols"] = ",".join(c for c in df.columns if c != "Date")
    frames.append(df)
print(f"downloaded {len(months)} months in {time.time() - t0:.0f}s; empty months: {empty_months}")

full = pd.concat(frames, ignore_index=True)
full.to_csv(OUT / f"{metric}_full_history.csv", index=False)
print("column sets seen:", full["_cols"].nunique())
for cs, g in full.groupby("_cols"):
    print(f"  {g.Date.min().date()} -> {g.Date.max().date()} ({len(g)} rows): {cs[:120]}...")
print("Id values:", full.Id.unique(), "| code values:", full.Values_code.unique())
print("duplicate dates:", full.Date.duplicated().sum())
all_days = pd.date_range(full.Date.min(), full.Date.max(), freq="D")
missing_days = all_days.difference(full.Date)
print("missing calendar days:", len(missing_days), list(missing_days[:20].date))
vals = full[hour_cols(full)].apply(pd.to_numeric, errors="coerce")
raw = full[hour_cols(full)]
print(
    "null/empty raw cells:",
    int(raw.isna().sum().sum() + (raw == "").sum().sum()),
    "| non-numeric cells:",
    int(vals.isna().sum().sum() - raw.isna().sum().sum()),
)
print("days with any NaN hour:", list(full.loc[vals.isna().any(axis=1), "Date"].dt.date)[:20])
print("non-positive cells:", int((vals <= 0).sum().sum()))
h = vals.stack()
print(
    f"hourly kWh: min={h.min():,.0f} p01={h.quantile(0.01):,.0f} median={h.median():,.0f} max={h.max():,.0f}"
)
daily = vals.sum(axis=1).set_axis(full.Date)
print("daily GWh by year (mean):")
print((daily.groupby(daily.index.year).mean() / 1e6).round(1).to_string())
print("lowest 8 days (GWh):")
print((daily.nsmallest(8) / 1e6).round(1).to_string())
