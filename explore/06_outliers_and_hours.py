"""Phase 0: hourly outliers in full history + hour-label convention via solar generation."""

import datetime as dt
from pathlib import Path

import pandas as pd
import requests
from xm_rest import fetch, hour_cols

OUT = Path(__file__).parent / "output"
pd.set_option("display.width", 250)

for metric in ["DemaCome", "DemaReal"]:
    full = pd.read_csv(OUT / f"{metric}_full_history.csv", parse_dates=["Date"])
    full = full[full.Date < "2026-11-06"]  # exclude known partial days
    wide = full.set_index("Date")[hour_cols(full)]
    wide.columns = range(1, 25)
    s = wide.stack().rename("kwh")
    s.index.names = ["date", "hour"]
    s = s.reset_index()
    # ratio to median of same hour over +-7 days
    s = s.sort_values(["hour", "date"])
    s["ref"] = s.groupby("hour")["kwh"].transform(
        lambda x: x.rolling(15, center=True, min_periods=5).median()
    )
    s["ratio"] = s.kwh / s.ref
    print(
        f"\n== {metric}: hours deviating >35% from same-hour 15-day rolling median: {(abs(s.ratio - 1) > 0.35).sum()}"
    )
    print(
        s[abs(s.ratio - 1) > 0.35]
        .sort_values("date")
        .assign(
            kwh=lambda d: d.kwh.round(0),
            ref=lambda d: d.ref.round(0),
            ratio=lambda d: d.ratio.round(2),
        )
        .to_string(index=False)
    )

# Hour convention: solar plants' generation by hour
lst = requests.post(
    "https://servapibi.xm.com.co/lists",
    json={"MetricId": "ListadoRecursos", "Entity": "Sistema"},
    timeout=60,
).json()
rec = pd.json_normalize(lst["Items"], "ListEntities", "Date", sep="_")
print("\nListadoRecursos columns:", list(rec.columns))
src_col = [c for c in rec.columns if "EnerSource" in c or "Fuel" in c or "Type" in c]
print(
    "source-like cols:",
    src_col,
    rec[src_col[0]].value_counts().head(10).to_dict() if src_col else "",
)
solar_codes = set(
    rec.loc[rec[src_col[0]].str.contains("RAD|SOL", case=False, na=False), "Values_Code"]
)
print("n solar resources:", len(solar_codes))
day = dt.date(2026, 11, 3)
g = fetch("Gene", "Recurso", day, day)
g = g[g.Values_code.isin(solar_codes)]
prof = g[hour_cols(g)].apply(pd.to_numeric).sum() / 1e3
prof.index = [c.replace("Values_Hour", "Hour") for c in prof.index]
print(
    f"\nSolar generation (MWh) by hour label on {day} ({g.Values_code.nunique()} plants reporting):"
)
print(prof.round(1).to_string())
