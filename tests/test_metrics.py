import pandas as pd
import pytest

from demand_forecast.daytypes import SATURDAY, SUNDAY_OR_HOLIDAY, WORKDAY, day_types
from demand_forecast.models import metrics

BOGOTA = "America/Bogota"


class TestDayTypes:
    @pytest.mark.parametrize(
        ("date", "expected"),
        [
            ("2026-11-11", WORKDAY),  # Wednesday; the holiday of this date moves to Monday 16
            ("2026-11-14", SATURDAY),
            ("2026-11-15", SUNDAY_OR_HOLIDAY),
            ("2026-11-16", SUNDAY_OR_HOLIDAY),  # Monday, Independence of Cartagena (moved)
            ("2026-04-03", SUNDAY_OR_HOLIDAY),  # Good Friday
            ("2022-01-01", SUNDAY_OR_HOLIDAY),  # New Year's Day on a Saturday
            ("2026-12-24", WORKDAY),
        ],
    )
    def test_known_days(self, date, expected):
        assert day_types(pd.DatetimeIndex([date], tz=BOGOTA)).iloc[0] == expected

    def test_result_is_indexed_by_the_given_days(self):
        days = pd.date_range("2026-11-13", periods=4, freq="D", tz=BOGOTA)
        types = day_types(days)
        assert types.index.equals(days)
        assert list(types) == [WORKDAY, SATURDAY, SUNDAY_OR_HOLIDAY, SUNDAY_OR_HOLIDAY]


@pytest.fixture
def results() -> pd.DataFrame:
    """Friday 13 to Monday 16 (holiday) of November 2026; actual 100, error set per day."""
    index = pd.date_range("2026-11-13", periods=4 * 24, freq="h", tz=BOGOTA)
    error_by_day = {13: 10.0, 14: 20.0, 15: -30.0, 16: 50.0}
    predicted = [100.0 + error_by_day[ts.day] for ts in index]
    return pd.DataFrame({"actual": 100.0, "predicted": predicted}, index=index)


def test_overall_mae_and_mape(results):
    summary = metrics.overall(results)
    assert summary["mae"] == pytest.approx(27.5)
    assert summary["mape"] == pytest.approx(27.5)


def test_mape_is_relative_to_the_actual_value():
    index = pd.date_range("2026-11-13", periods=2, freq="h", tz=BOGOTA)
    results = pd.DataFrame({"actual": [200.0, 50.0], "predicted": [210.0, 45.0]}, index=index)
    summary = metrics.overall(results)
    assert summary["mae"] == pytest.approx(7.5)
    assert summary["mape"] == pytest.approx((5.0 + 10.0) / 2)


def test_by_day_type_groups_sundays_with_holidays(results):
    table = metrics.by_day_type(results)
    assert list(table.index) == [WORKDAY, SATURDAY, SUNDAY_OR_HOLIDAY]
    assert table.loc[WORKDAY, "mape"] == pytest.approx(10.0)
    assert table.loc[SATURDAY, "mape"] == pytest.approx(20.0)
    assert table.loc[SUNDAY_OR_HOLIDAY, "mape"] == pytest.approx(40.0)


def test_by_month_labels_each_calendar_month():
    index = pd.date_range("2026-10-31", periods=48, freq="h", tz=BOGOTA)
    predicted = [110.0] * 24 + [130.0] * 24
    table = metrics.by_month(pd.DataFrame({"actual": 100.0, "predicted": predicted}, index=index))
    assert list(table.index) == ["2026-10", "2026-11"]
    assert list(table["mape"]) == pytest.approx([10.0, 30.0])


def test_by_day_gives_the_error_and_the_type_of_each_day(results):
    table = metrics.by_day(results)
    assert list(table.index) == list(pd.date_range("2026-11-13", periods=4, freq="D", tz=BOGOTA))
    assert list(table["mape"]) == pytest.approx([10.0, 20.0, 30.0, 50.0])
    assert list(table["day_type"]) == [WORKDAY, SATURDAY, SUNDAY_OR_HOLIDAY, SUNDAY_OR_HOLIDAY]


def test_by_hour_uses_the_local_hour(results):
    results.loc[results.index.hour == 19, "predicted"] = 100.0
    table = metrics.by_hour(results)
    assert list(table.index) == list(range(24))
    assert table.loc[19, "mae"] == 0.0
    assert table.loc[18, "mae"] == pytest.approx(27.5)
