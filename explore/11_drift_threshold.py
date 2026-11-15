"""Phase 4: how large the drift score is in ordinary times.

The drift report compares the features of the last 28 forecast days with the ten years the
model was trained on, with the Wasserstein distance divided by the standard deviation of the
reference (Evidently's "normed" distance). This script computes that same score for every
28-day window inside the training period, to see what a window with nothing wrong looks like.

Usage (no network, about a minute):
    python explore/11_drift_threshold.py
"""

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from demand_forecast import config
from demand_forecast.data import store
from demand_forecast.models.backtest import History
from demand_forecast.models.registry import registry_entry
from demand_forecast.monitoring.drift import DRIFT_FEATURES, reference_features

WINDOW_HOURS = config.DRIFT_WINDOW_DAYS * 24
STEP_HOURS = 7 * 24
QUANTILES = [0.50, 0.90, 0.95, 0.99, 1.00]


def main() -> None:
    history = History(
        demand=store.load_series(config.DEMAND.name),
        generation=store.load_series(config.GENERATION.name),
    )
    reference = reference_features(history, registry_entry())
    print(
        f"Reference {reference.index[0].date()}..{reference.index[-1].date()}, "
        f"{config.DRIFT_WINDOW_DAYS}-day windows every 7 days.\n"
    )
    rows = {}
    for name in DRIFT_FEATURES:
        values = reference[name].to_numpy()
        scale = max(values.std(), 0.001)
        scores = [
            wasserstein_distance(values, values[start : start + WINDOW_HOURS]) / scale
            for start in range(0, len(values) - WINDOW_HOURS + 1, STEP_HOURS)
        ]
        rows[name] = np.quantile(scores, QUANTILES)
        windows = len(scores)
    table = pd.DataFrame(rows, index=[f"p{round(q * 100)}" for q in QUANTILES]).T.round(2)
    print(table.to_string())
    print(f"\n{windows} windows per feature.")
    for threshold in (0.1, 0.3, 0.5, 0.75, 1.0):
        flagged = [
            np.mean(
                [
                    wasserstein_distance(values, values[start : start + WINDOW_HOURS])
                    / max(values.std(), 0.001)
                    >= threshold
                    for start in range(0, len(values) - WINDOW_HOURS + 1, STEP_HOURS)
                ]
            )
            for values in (reference[name].to_numpy() for name in DRIFT_FEATURES)
        ]
        print(f"threshold {threshold}: {np.mean(flagged):.1%} of feature-windows flagged")


if __name__ == "__main__":
    main()
