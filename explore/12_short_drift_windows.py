"""Phase 4: how often an ordinary window shorter than a week counts as drifted.

The drift thresholds come from windows of the reference taken every 7 days. For a window
shorter than a week those all start on the same weekday, so the threshold only knows that
weekday. This script takes windows starting on every day of the last two years of the training
period, scores them against the thresholds the drift report would use and counts how many
have at least one drifted feature, by the weekday they start on.

Usage (no network, about 12 minutes):
    python explore/12_short_drift_windows.py
"""

import numpy as np
from scipy.stats import wasserstein_distance

from demand_forecast import config
from demand_forecast.data import store
from demand_forecast.models.backtest import History
from demand_forecast.models.registry import registry_entry
from demand_forecast.monitoring.drift import (
    DRIFT_FEATURES,
    MIN_SCALE,
    reference_features,
    window_thresholds,
)

WINDOW_DAYS = [1, 2, 3, 7, 28]
CHECKED_DAYS = 2 * 365


def main() -> None:
    history = History(
        demand=store.load_series(config.DEMAND.name),
        generation=store.load_series(config.GENERATION.name),
    )
    reference = reference_features(history, registry_entry())
    values = {name: reference[name].to_numpy() for name in DRIFT_FEATURES}
    scales = {name: max(column.std(), MIN_SCALE) for name, column in values.items()}
    first = len(reference) - CHECKED_DAYS * 24
    print(
        f"Reference {reference.index[0].date()}..{reference.index[-1].date()}, starting on a "
        f"{reference.index[0].day_name()}; windows from {reference.index[first].date()} on.\n"
    )
    for days in WINDOW_DAYS:
        hours = days * 24
        thresholds = window_thresholds(reference, hours)
        starts = range(first, len(reference) - hours + 1, 24)
        flagged = np.array(
            [
                any(
                    wasserstein_distance(values[name], values[name][start : start + hours])
                    / scales[name]
                    >= thresholds[name]
                    for name in DRIFT_FEATURES
                )
                for start in starts
            ]
        )
        weekdays = reference.index[list(starts)].dayofweek.to_numpy()
        by_weekday = "  ".join(f"{flagged[weekdays == number].mean():4.0%}" for number in range(7))
        print(
            f"{days:2d}-day windows ({len(starts)}): {flagged.mean():5.1%} with a drifted feature; "
            f"by start weekday, Monday to Sunday: {by_weekday}"
        )


if __name__ == "__main__":
    main()
