"""Phase 0: download recent hourly demand (DemaReal, DemaCome @ Sistema), inspect format and lag.

Needs pydataxm, which is no longer a project dependency:
    uv run --with pydataxm python explore/02_recent_sample.py
"""

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import requests
from pydataxm.pydataxm import ReadDB

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)

today = dt.date.today()
start = today - dt.timedelta(days=21)
print("run at (local):", dt.datetime.now().isoformat(timespec="seconds"), "| today:", today)

# Raw REST call, bypassing pydataxm
body = {
    "MetricId": "DemaReal",
    "StartDate": str(start),
    "EndDate": str(today),
    "Entity": "Sistema",
    "Filter": [],
}
r = requests.post("https://servapibi.xm.com.co/hourly", json=body, timeout=60)
print("REST status:", r.status_code, "| content-type:", r.headers.get("content-type"))
raw = r.json()
(OUT / f"raw_DemaReal_{today}.json").write_text(json.dumps(raw, indent=1, ensure_ascii=False))
print("top-level keys:", list(raw.keys()))
print("n items:", len(raw["Items"]))
print("first item:", json.dumps(raw["Items"][0], ensure_ascii=False)[:900])

api = ReadDB()
for metric in ["DemaReal", "DemaCome"]:
    df = api.request_data(metric, "Sistema", start, today)
    df.to_csv(OUT / f"{metric}_recent_{today}.csv", index=False)
    print(f"\n===== {metric} | shape={df.shape}")
    print(df.dtypes.to_string())
    print(df.head(3).to_string())
    print("Date range:", df.Date.min(), "->", df.Date.max())
    hour_cols = [c for c in df.columns if c.startswith("Values_Hour")]
    vals = df[hour_cols].apply(pd.to_numeric, errors="coerce")
    print("hour cols:", len(hour_cols), hour_cols[:2], "...", hour_cols[-2:])
    print("NaN per day (last 10):")
    print(
        pd.concat([df.Date, vals.isna().sum(axis=1).rename("n_nan")], axis=1).tail(10).to_string()
    )
    print("daily totals GWh (last 10):")
    print(pd.concat([df.Date, (vals.sum(axis=1) / 1e6).rename("GWh")], axis=1).tail(10).to_string())
