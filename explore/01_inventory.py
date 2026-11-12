"""Phase 0: list XM API metric inventory and locate demand metrics.

Needs pydataxm, which is no longer a project dependency:
    uv run --with pydataxm python explore/01_inventory.py
"""

from pathlib import Path

import pandas as pd
from pydataxm.pydataxm import ReadDB

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 80)

api = ReadDB()
inv = api.get_collections()
inv.to_csv(OUT / "inventory.csv", index=False)
print("columns:", list(inv.columns))
print("rows:", len(inv), "unique MetricId:", inv.MetricId.nunique())
print(inv.Type.value_counts())

mask = inv.MetricId.str.contains("Dema", case=False) | inv.MetricName.str.contains(
    "demanda", case=False
)
print(inv[mask].sort_values(["MetricId", "Entity"]).to_string())
