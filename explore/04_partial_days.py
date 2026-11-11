"""Phase 0: characterise the partial most-recent days and compare related metrics."""

import datetime as dt

import pandas as pd

from xm_rest import daily_total, fetch, hour_cols

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)

today = dt.date.today()
start = today - dt.timedelta(days=12)
print("run at:", dt.datetime.now().isoformat(timespec="seconds"))

totals = {}
for m in ["DemaReal", "DemaCome", "DemaRealReg", "DemaRealNoReg", "Gene", "PerdidasEner"]:
    df = fetch(m, "Sistema", start, today)
    totals[m] = daily_total(df) / 1e6 if not df.empty else pd.Series(dtype=float)
d = fetch("DemaSIN", "Sistema", start, today, period="daily")
print("DemaSIN raw columns:", list(d.columns))
totals["DemaSIN(daily)"] = pd.to_numeric(d["Value"]).groupby(d["Date"]).sum() / 1e6
print("\nDaily totals (GWh):")
print(pd.DataFrame(totals).round(2).to_string())

# Hourly profile of last available days for DemaReal
dr = fetch("DemaReal", "Sistema", today - dt.timedelta(days=6), today)
prof = dr.set_index(dr.Date.dt.date)[hour_cols(dr)].apply(pd.to_numeric) / 1e6
prof.columns = [c.replace("Values_Hour", "h") for c in prof.columns]
print("\nDemaReal hourly (GWh) last days:")
print(prof.round(2).T.to_string())

# How many agents report on each day?
ag = fetch("DemaReal", "Agente", today - dt.timedelta(days=6), today)
print("\nDemaReal Agente: n agents per day")
print(ag.groupby("Date")["Values_code"].nunique().to_string())
