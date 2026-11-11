"""Phase 0: snapshot recent data to measure publication lag and detect revisions.

Usage:
    uv run python explore/08_snapshot.py          # take a snapshot now
    uv run python explore/08_snapshot.py compare  # diff the two latest snapshots (revisions)
    uv run python explore/08_snapshot.py summary  # lag (D-k) of every snapshot taken so far
"""

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
from xm_rest import fetch, hour_cols

SNAP = Path(__file__).parent / "output" / "snapshots"
SNAP.mkdir(parents=True, exist_ok=True)
METRICS = ["DemaCome", "DemaReal", "Gene"]


def completeness(df: pd.DataFrame) -> pd.DataFrame:
    tot = df.set_index("Date")[hour_cols(df)].apply(pd.to_numeric).sum(axis=1) / 1e6
    ref = tot.rolling(28, min_periods=7).median().shift(1)
    return pd.DataFrame({"GWh": tot.round(2), "ratio_vs_28d_median": (tot / ref).round(3)})


def take() -> None:
    now = dt.datetime.now()
    today = now.date()
    for m in METRICS:
        df = fetch(m, "Sistema", today - dt.timedelta(days=31), today)
        df["fetched_at"] = now.isoformat(timespec="seconds")
        df.to_csv(SNAP / f"{m}_{now:%Y%m%dT%H%M}.csv", index=False)
        c = completeness(df)
        complete = c[c.ratio_vs_28d_median > 0.9].index.max().date()
        print(
            f"{now:%Y-%m-%d %H:%M} {m:9s} last date={df.Date.max().date()} "
            f"(D-{(today - df.Date.max().date()).days})  last complete={complete} (D-{(today - complete).days})"
        )
        print(c.tail(4).to_string())


def compare() -> None:
    for m in METRICS:
        files = sorted(SNAP.glob(f"{m}_*.csv"))
        if len(files) < 2:
            print(f"{m}: need two snapshots")
            continue
        a, b = (pd.read_csv(f, parse_dates=["Date"]) for f in files[-2:])
        common = sorted(set(a.Date) & set(b.Date))
        va = a.set_index("Date").loc[common, hour_cols(a)].apply(pd.to_numeric)
        vb = b.set_index("Date").loc[common, hour_cols(b)].apply(pd.to_numeric)
        changed = (va - vb).abs() > 0.5
        print(f"\n{m}: {files[-2].stem} vs {files[-1].stem}")
        print(f"  cells changed: {int(changed.values.sum())} / {changed.size}; days with changes:")
        rel = ((vb.sum(axis=1) - va.sum(axis=1)) / va.sum(axis=1) * 100).round(3)
        print(rel[changed.any(axis=1)].rename("daily_total_change_%").to_string())


def summary() -> None:
    """One line per snapshot: weekday, last date returned and last complete date, as D-k."""
    rows = []
    for f in sorted(SNAP.glob("*.csv")):
        df = pd.read_csv(f, parse_dates=["Date"])
        taken = pd.Timestamp(df["fetched_at"].iloc[0])
        c = completeness(df)
        complete = c[c.ratio_vs_28d_median > 0.9].index.max()
        rows.append(
            {
                "metric": f.stem.split("_")[0],
                "taken": taken.strftime("%Y-%m-%d %a %H:%M"),
                "last_returned": f"D-{(taken.normalize() - df.Date.max()).days}",
                "last_complete": f"{complete:%a %m-%d} (D-{(taken.normalize() - complete).days})",
            }
        )
    print(pd.DataFrame(rows).sort_values(["metric", "taken"]).to_string(index=False))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "take"
    {"take": take, "compare": compare, "summary": summary}[mode]()
