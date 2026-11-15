"""Phase 4: how much the forecast degrades when XM publishes late.

Backtest on the selection window (the 12 months before the evaluation window, so this never
touches the evaluation). The model is fitted as configured, with the usual lags, but every
forecast only sees data that is `extra` days older than usual, as if XM had not published for
`extra` days. The baselines get the same late data, for reference.

Usage (no network, about 30 seconds per row):
    python explore/10_stale_data.py
"""

import pandas as pd

from demand_forecast import config
from demand_forecast.data import store
from demand_forecast.models import metrics
from demand_forecast.models.backtest import evaluation_window, make_folds, run_backtest
from demand_forecast.models.baselines import BASELINES
from demand_forecast.models.train import LightGBMForecaster
from demand_forecast.models.tune import selection_window

EXTRA_DAYS = [0, 1, 2, 3, 5, 7, 10]


def main() -> None:
    demand = store.load_series(config.DEMAND.name)
    generation = store.load_series(config.GENERATION.name)
    evaluation_start, _ = evaluation_window(demand.index.max().normalize())
    start, end = selection_window(evaluation_start)
    folds = make_folds(start, end)
    print(f"Selection window {start.date()}..{end.date()}, {len(folds)} folds. MAPE in %.\n")

    rows = []
    for extra in EXTRA_DAYS:
        lags = {
            "lag_days": config.DATA_LAG_DAYS + extra,
            "generation_lag_days": config.GENERATION_LAG_DAYS + extra,
        }
        row = {"extra_days": extra, "demand_cut": f"D-{lags['lag_days']}"}
        models = {name: build() for name, build in BASELINES.items()}
        models[LightGBMForecaster.name] = LightGBMForecaster()
        for name, model in models.items():
            results = run_backtest(demand, model, folds, generation=generation, **lags)
            row[name] = round(metrics.overall(results)["mape"], 3)
        rows.append(row)
        print(row, flush=True)

    print("\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
