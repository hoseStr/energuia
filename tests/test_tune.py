import numpy as np
import pandas as pd
import pytest

from demand_forecast import config
from demand_forecast.models.backtest import evaluation_window, make_folds
from demand_forecast.models.tune import build_candidate, candidates, selection_window, tune

BOGOTA = config.TIMEZONE


def day(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=BOGOTA)


class TestSelectionWindow:
    def test_is_the_twelve_months_right_before_the_evaluation_window(self):
        evaluation_start, _ = evaluation_window(day("2026-11-08"))
        assert selection_window(evaluation_start) == (day("2024-11-01"), day("2025-10-31"))

    def test_never_overlaps_the_evaluation_window(self):
        for last_day in pd.date_range("2026-01-01", "2026-12-31", freq="17D", tz=BOGOTA):
            evaluation_start, _ = evaluation_window(last_day)
            _, selection_end = selection_window(evaluation_start)
            assert selection_end < evaluation_start


class TestCandidates:
    def test_every_combination_of_the_grid(self):
        grid = {"train_years": [3, 5], "num_leaves": [15, 31, 63]}
        found = candidates(grid)
        assert len(found) == 6
        assert found[0] == {"train_years": 3, "num_leaves": 15}
        assert found[-1] == {"train_years": 5, "num_leaves": 63}

    def test_forecaster_settings_and_lightgbm_parameters_go_to_their_place(self):
        model = build_candidate({"train_years": 3, "rounds": 50, "num_leaves": 7})
        assert model.train_years == 3
        assert model.rounds == 50
        assert model.params == {**config.LGBM_PARAMS, "num_leaves": 7}

    def test_values_missing_from_a_candidate_stay_as_configured(self):
        model = build_candidate({"num_leaves": 7})
        assert model.train_years == config.TRAIN_YEARS
        assert model.rounds == config.LGBM_ROUNDS

    def test_the_configured_model_is_one_of_the_candidates_of_the_grid(self):
        configured = {
            "train_years": config.TRAIN_YEARS,
            "rounds": config.LGBM_ROUNDS,
            **{
                name: config.LGBM_PARAMS[name]
                for name in config.TUNING_GRID
                if name in config.LGBM_PARAMS
            },
        }
        assert configured in candidates(config.TUNING_GRID)


def test_tune_scores_every_candidate_and_sorts_them_by_mape(make_series):
    rng = np.random.default_rng(11)
    clean = make_series("2025-06-01", 300)
    demand = clean * rng.normal(1, 0.03, len(clean))
    generation = clean * rng.normal(1, 0.03, len(clean))
    folds = make_folds(day("2026-02-01"), day("2026-02-14"))
    seen = []

    table = tune(
        demand,
        generation,
        folds,
        grid={"rounds": [1, 30], "num_leaves": [4]},
        on_result=seen.append,
    )

    assert list(table.columns) == ["rounds", "num_leaves", "mae", "mape"]
    assert len(table) == len(seen) == 2
    assert [row["rounds"] for row in seen] == [1, 30]
    assert list(table["mape"]) == sorted(table["mape"])
    assert table["mape"].iloc[0] == pytest.approx(min(row["mape"] for row in seen))
