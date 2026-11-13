"""LightGBM forecaster: one model for the 24 hours, with the target hour as a feature."""

from collections.abc import Mapping
from dataclasses import dataclass

import lightgbm as lgb
import pandas as pd

from demand_forecast import config
from demand_forecast.features.build import (
    FEATURES,
    SCALE,
    TARGET,
    build_features,
    training_frame,
)
from demand_forecast.models.backtest import History

HOURS_PER_DAY = 24


@dataclass(frozen=True)
class TrainingData:
    """The target days a model learned from."""

    first_target_day: pd.Timestamp
    last_target_day: pd.Timestamp
    rows: int


class LightGBMForecaster:
    """Forecast the demand relative to the recent profile of the target's day type."""

    name = "lightgbm"

    def __init__(
        self,
        params: Mapping[str, object] = config.LGBM_PARAMS,
        *,
        rounds: int = config.LGBM_ROUNDS,
        train_years: int = config.TRAIN_YEARS,
        horizon_days: int = config.HORIZON_DAYS,
        lag_days: int = config.DATA_LAG_DAYS,
        generation_lag_days: int = config.GENERATION_LAG_DAYS,
        booster: lgb.Booster | None = None,
    ) -> None:
        self.params = dict(params)
        self.rounds = rounds
        self.train_years = train_years
        self.horizon_days = horizon_days
        self.lag_days = lag_days
        self.generation_lag_days = generation_lag_days
        self.booster = booster
        self.training_data: TrainingData | None = None

    def fit(self, history: History) -> None:
        """Learn from the last `train_years` years of published days."""
        if history.demand.empty:
            raise ValueError("no published history to learn from")
        last_day = history.demand.index[-1].normalize()
        frame = training_frame(
            history,
            horizon_days=self.horizon_days,
            lag_days=self.lag_days,
            generation_lag_days=self.generation_lag_days,
            first_target=last_day - pd.DateOffset(years=self.train_years) + pd.Timedelta(days=1),
        )
        if frame.empty:
            raise ValueError("the published history is too short to learn from")
        self.booster = lgb.train(
            self.params, lgb.Dataset(frame[FEATURES], frame[TARGET]), num_boost_round=self.rounds
        )
        self.training_data = TrainingData(
            first_target_day=frame.index[0].normalize(),
            last_target_day=frame.index[-1].normalize(),
            rows=len(frame),
        )

    def predict(self, history: History, target_day: pd.Timestamp) -> pd.Series:
        """Return the 24 hourly values forecast for `target_day`.

        The features use each series up to the last day `history` holds, whatever its lag.
        """
        if self.booster is None:
            raise ValueError("the model has not been fitted")
        if history.generation is None or history.generation.empty or history.demand.empty:
            raise ValueError(f"no published demand and generation to forecast {target_day.date()}")
        demand = history.demand.iloc[-config.FEATURE_WINDOW_DAYS * HOURS_PER_DAY :]
        generation = history.generation.iloc[-2 * HOURS_PER_DAY :]
        rows = build_features(
            demand,
            generation,
            pd.DatetimeIndex([target_day]),
            demand_cut=pd.DatetimeIndex([demand.index[-1].normalize()]),
            generation_cut=pd.DatetimeIndex([generation.index[-1].normalize()]),
        )
        return pd.Series(self.booster.predict(rows[FEATURES]) * rows[SCALE], index=rows.index)
