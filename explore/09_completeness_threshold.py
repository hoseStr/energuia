"""Phase 0: which threshold separates partial days from complete ones? (offline)

Reads the CSVs written by 05_history.py (DemaCome) and 07_gene_vs_demand.py (Gene).
"""

from pathlib import Path

import pandas as pd
from xm_rest import hour_cols

OUT = Path(__file__).parent / "output"
LAST_COMPLETE = "2026-11-05"  # last complete DemaCome day in the 2026-11-10 download
THRESHOLDS = (0.9, 0.85, 0.8, 0.75, 0.7, 0.6)


def daily_gwh(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.set_index("Date")[hour_cols(df)].apply(pd.to_numeric).sum(axis=1).sort_index() / 1e6


dema = daily_gwh(OUT / "DemaCome_full_history.csv")
ratio = (dema / dema.rolling(28, min_periods=28).median().shift(1)).dropna()
complete, partial = ratio[:LAST_COMPLETE], ratio[LAST_COMPLETE:].iloc[1:]

print(f"Signal 1: daily total / median of the previous 28 days ({len(complete)} complete days)")
for th in THRESHOLDS:
    n = int((complete < th).sum())
    print(f"  ratio < {th:.2f}: {n:5d} complete days rejected ({n / len(complete):.1%})")
print("  by weekday, complete days under 0.90:")
print(complete[complete < 0.9].groupby(lambda d: d.day_name()).size().to_string())
print("  lowest complete days:")
print(complete.nsmallest(5).round(3).to_string())
print("  partial days:")
print(partial.round(3).to_string())

gene = daily_gwh(OUT / "Gene_last12m.csv")
cross = (dema / gene).dropna()
ok = cross[:LAST_COMPLETE]
print(f"\nSignal 2: DemaCome / Gene daily totals ({len(ok)} complete days)")
print(f"  complete days: min={ok.min():.4f} max={ok.max():.4f}")
print("  partial days:")
print(cross[LAST_COMPLETE:].iloc[1:].round(3).to_string())
