import numpy as np
import pandas as pd
import pytest

from demand_forecast import config
from demand_forecast.models.backtest import History, history_at
from demand_forecast.models.train import LightGBMForecaster
from demand_forecast.serving.forecast import issue_forecast

BOGOTA = config.TIMEZONE
VERSION = "20261113T163005Z"
# Saturday evening in Bogota. The stored demand ends on 2026-11-08 and the generation on
# 2026-11-10, as they did in the repo that day.
NOW = pd.Timestamp("2026-11-14 18:08", tz=BOGOTA)


def day(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=BOGOTA)


@pytest.fixture(scope="module")
def series() -> tuple[pd.Series, pd.Series]:
    """Noisy synthetic demand up to 2026-11-10; the same days of generation, with other noise."""
    rng = np.random.default_rng(11)
    index = pd.date_range(day("2026-01-01"), day("2026-11-10 23:00"), freq="h")
    daily_shape = 1 + 0.2 * np.sin(index.hour / 24 * 2 * np.pi)
    weekly_shape = np.where(index.dayofweek == 6, 0.85, 1.0)
    clean = pd.Series(1e7 * daily_shape * weekly_shape, index=index)
    return clean * rng.normal(1, 0.03, len(clean)), clean * rng.normal(1, 0.03, len(clean))


@pytest.fixture(scope="module")
def model(series) -> LightGBMForecaster:
    demand, generation = series
    fitted = LightGBMForecaster(rounds=20, train_years=1)
    fitted.fit(history_at(demand, generation, day("2026-10-01")))
    return fitted


@pytest.fixture
def published(series) -> History:
    """What the repo held on 2026-11-14: demand up to D-6 and generation up to D-4."""
    demand, generation = series
    return History(
        demand=demand[demand.index < day("2026-11-09")],
        generation=generation[generation.index < day("2026-11-11")],
    )


def test_the_forecast_is_for_the_day_after_it_is_issued(model, published):
    forecast = issue_forecast(model, VERSION, published, now=NOW)

    assert forecast.issued_at == NOW
    assert forecast.target_day == day("2026-11-15")
    assert list(forecast.values.index) == list(
        pd.date_range(day("2026-11-15"), periods=24, freq="h")
    )
    assert forecast.model_version == VERSION


def test_the_forecast_says_how_recent_its_data_is(model, published):
    forecast = issue_forecast(model, VERSION, published, now=NOW)

    assert forecast.demand_last_day == day("2026-11-08")
    assert forecast.generation_last_day == day("2026-11-10")


def test_the_values_are_the_prediction_of_the_model_on_everything_published(model, published):
    forecast = issue_forecast(model, VERSION, published, now=NOW)

    expected = model.predict(published, day("2026-11-15"))
    assert list(forecast.values) == list(expected)
    assert (forecast.values > 0).all()


def test_the_issue_day_is_the_bogota_date_whatever_the_timezone_of_the_clock(model, published):
    # 02:00 UTC on the 15th is still 21:00 of the 14th in Bogota.
    utc_now = pd.Timestamp("2026-11-15 02:00", tz="UTC")

    forecast = issue_forecast(model, VERSION, published, now=utc_now)

    assert forecast.target_day == day("2026-11-15")
    assert forecast.issued_at == utc_now
    assert str(forecast.issued_at.tz) == BOGOTA


def test_a_clock_without_timezone_is_rejected(model, published):
    with pytest.raises(ValueError, match="timezone-aware"):
        issue_forecast(model, VERSION, published, now=pd.Timestamp("2026-11-14 18:08"))


@pytest.mark.parametrize("stale", ["demand", "generation"])
def test_data_of_days_that_were_not_over_at_issue_time_is_rejected(model, series, published, stale):
    """A live forecast can never hold the issue day: seeing it means the clock is wrong."""
    demand, generation = series
    earlier = pd.Timestamp("2026-11-10 09:00", tz=BOGOTA)
    history = History(
        demand=demand if stale == "demand" else published.demand,
        generation=generation if stale == "generation" else published.generation,
    )

    with pytest.raises(ValueError, match="2026-11-10 was not over"):
        issue_forecast(model, VERSION, history, now=earlier)


def test_without_generation_there_is_no_forecast(model, published):
    with pytest.raises(ValueError, match="no published demand and generation"):
        issue_forecast(model, VERSION, History(demand=published.demand), now=NOW)
