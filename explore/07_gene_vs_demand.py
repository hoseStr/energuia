"""Phase 0: how closely does hourly Gene (Sistema) track DemaCome / DemaReal over the last 12 months?"""

import datetime as dt
from pathlib import Path

import pandas as pd

from xm_rest import fetch, hour_cols

OUT = Path(__file__).parent / "output"


def long(df: pd.DataFrame) -> pd.Series:
    w = df.set_index("Date")[hour_cols(df)].apply(pd.to_numeric)
    w.columns = range(24)
    s = w.stack()
    return pd.Series(s.values, index=s.index.get_level_values(0) + pd.to_timedelta(s.index.get_level_values(1), "h"))


months = pd.date_range("2025-11-01", "2026-11-01", freq="MS")
gene = pd.concat([fetch("Gene", "Sistema", m.date(), (m + pd.offsets.MonthEnd(0)).date()) for m in months])
gene.to_csv(OUT / "Gene_last12m.csv", index=False)
g = long(gene)
for metric in ["DemaCome", "DemaReal"]:
    full = pd.read_csv(OUT / f"{metric}_full_history.csv", parse_dates=["Date"])
    d = long(full[(full.Date >= "2025-11-01") & (full.Date < "2026-11-06")])
    both = pd.concat([g.rename("gene"), d.rename("dem")], axis=1).dropna()
    rel = (both.gene - both.dem) / both.dem * 100
    print(f"Gene vs {metric} (hourly, n={len(both)}): mean diff={rel.mean():.2f}%  "
          f"p05={rel.quantile(.05):.2f}%  p95={rel.quantile(.95):.2f}%  max|diff|={rel.abs().max():.2f}%  corr={both.corr().iloc[0, 1]:.5f}")
print("Gene last complete-looking day:", gene.Date.max().date())
