import json

import numpy as np
import pandas as pd
import pytest

from demand_forecast import config
from demand_forecast.data import store
from demand_forecast.monitoring.drift import DriftSummary
from demand_forecast.monitoring.report import build_payload, render, write_panel
from demand_forecast.serving.forecast import Forecast
from demand_forecast.serving.log import append_forecast, read_log

BOGOTA = config.TIMEZONE
ONE_DAY = pd.Timedelta(days=1)
NOW = pd.Timestamp("2026-11-25 15:20", tz=BOGOTA)
ENTRY = {
    "version": "20261113T215929Z",
    "backtest": {
        "start": "2025-11-01",
        "end": "2026-10-31",
        "folds": 12,
        "metrics": {
            "seasonal-naive": {"mae": 338628.4, "mape": 3.4025},
            "moving-average": {"mae": 300503.6, "mape": 3.034},
            "lightgbm": {"mae": 194519.8, "mape": 1.9585},
        },
    },
}


def day(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=BOGOTA)


@pytest.fixture
def demand() -> pd.Series:
    """Actual demand published up to 2026-11-19: 1e7 kWh in every hour."""
    index = pd.date_range(day("2026-10-01"), day("2026-11-19 23:00"), freq="h")
    return pd.Series(1e7, index=index)


def log_days(data_dir, first: str, count: int, error: float = 0.02) -> pd.DataFrame:
    """Log `count` consecutive days, each forecast 2 % above 1e7 kWh in every hour."""
    for offset in range(count):
        target_day = day(first) + offset * ONE_DAY
        append_forecast(
            Forecast(
                issued_at=target_day - ONE_DAY + pd.Timedelta(hours=15, minutes=17),
                target_day=target_day,
                model_version="20261113T215929Z",
                demand_last_day=target_day - 6 * ONE_DAY,
                generation_last_day=target_day - 4 * ONE_DAY,
                values=pd.Series(
                    1e7 * (1 + error), index=pd.date_range(target_day, periods=24, freq="h")
                ),
            ),
            data_dir,
        )
    return read_log(data_dir)


def test_the_payload_has_the_live_error_and_the_days(tmp_path, demand):
    log = log_days(tmp_path, "2026-11-15", 11)

    payload = build_payload(log, demand, ENTRY, now=NOW)

    assert payload["model_version"] == "20261113T215929Z"
    assert payload["generated_at"] == "2026-11-25T15:20:00-05:00"
    assert payload["backtest"] == {
        "start": "2025-11-01",
        "end": "2026-10-31",
        "mape": 1.958,
        "best_baseline_mape": 3.034,
    }
    assert payload["alert_threshold"] == pytest.approx(1.5 * 1.9585, abs=1e-3)
    assert payload["live"] == {
        "evaluated_days": 5,
        "first_day": "2026-11-15",
        "last_day": "2026-11-19",
        "mape": 2.0,
        "recent_days": 7,
        "recent_mape": None,
    }
    assert payload["latest"] == {
        "target_day": "2026-11-25",
        "issued_at": "2026-11-24T15:17:00-05:00",
        "total_gwh": 244.8,
    }
    assert len(payload["days"]) == 11
    assert payload["days"][0] == {
        "day": "2026-11-15",
        "day_type": "sunday_or_holiday",
        "predicted_gwh": 244.8,
        "actual_gwh": 240.0,
        "mape": 2.0,
        "cumulative_mape": 2.0,
    }
    assert payload["days"][-1] == {
        "day": "2026-11-25",
        "day_type": "workday",
        "predicted_gwh": 244.8,
        "actual_gwh": None,
        "mape": None,
        "cumulative_mape": None,
    }


def test_the_hourly_series_ends_its_actuals_where_the_data_ends(tmp_path, demand):
    log = log_days(tmp_path, "2026-11-15", 11)

    hourly = build_payload(log, demand, ENTRY, now=NOW)["hourly"]

    assert len(hourly) == 11 * 24
    assert hourly[0] == ["2026-11-15T00:00:00-05:00", 10.2, 10.0]
    assert hourly[5 * 24 - 1] == ["2026-11-19T23:00:00-05:00", 10.2, 10.0]
    assert hourly[5 * 24] == ["2026-11-20T00:00:00-05:00", 10.2, None]


def test_the_hourly_series_only_covers_the_last_days(tmp_path, demand):
    log = log_days(tmp_path, "2026-11-15", 11)

    payload = build_payload(log, demand, ENTRY, now=NOW, days=3)

    assert len(payload["hourly"]) == 3 * 24
    assert payload["hourly"][0][0] == "2026-11-23T00:00:00-05:00"
    assert len(payload["days"]) == 11


def test_the_cumulative_error_is_the_running_mean(tmp_path, demand):
    log_days(tmp_path, "2026-11-15", 1, error=0.01)
    log = log_days(tmp_path, "2026-11-16", 1, error=0.03)

    days = build_payload(log, demand, ENTRY, now=NOW)["days"]

    assert [row["mape"] for row in days] == [1.0, 3.0]
    assert [row["cumulative_mape"] for row in days] == [1.0, 2.0]


def test_the_recent_error_appears_with_seven_evaluated_days(tmp_path, demand):
    log = log_days(tmp_path, "2026-11-10", 10)
    assert build_payload(log, demand, ENTRY, now=NOW)["live"]["recent_mape"] == 2.0


def test_an_empty_log_still_gives_a_page(tmp_path, demand):
    payload = build_payload(read_log(tmp_path), demand, ENTRY, now=NOW)

    assert payload["live"] is None
    assert payload["latest"] is None
    assert payload["hourly"] == []
    assert payload["days"] == []
    assert payload["drift"] is None
    assert payload["drift_min_days"] == config.DRIFT_MIN_DAYS
    assert "__PAYLOAD__" not in render(payload)


def test_the_page_embeds_the_payload_as_json(tmp_path, demand):
    payload = build_payload(log_days(tmp_path, "2026-11-15", 2), demand, ENTRY, now=NOW)

    page = render(payload)

    start = page.index('<script type="application/json" id="data">') + len(
        '<script type="application/json" id="data">'
    )
    embedded = page[start : page.index("</script>", start)]
    assert json.loads(embedded) == payload


def test_text_that_closes_a_script_cannot_break_out_of_the_data(tmp_path, demand):
    payload = build_payload(read_log(tmp_path), demand, ENTRY, now=NOW)
    payload["model_version"] = "</script><script>alert(1)</script>"

    page = render(payload)

    assert "</script><script>alert(1)" not in page


def test_the_drift_summary_travels_with_the_payload(tmp_path, demand):
    drift = DriftSummary(
        model_version="20261113T215929Z",
        reference_start="2016-11-09",
        reference_end="2026-11-08",
        current_start="2026-11-15",
        current_end="2026-11-25",
        current_days=11,
        method="Wasserstein distance (normed)",
        quantile=0.99,
        thresholds={"week_level": 0.25, "profile_shape": 0.1},
        scores={"week_level": 0.31, "profile_shape": 0.02},
        drifted=["week_level"],
    )

    payload = build_payload(read_log(tmp_path), demand, ENTRY, now=NOW, drift=drift)

    assert payload["drift"]["drifted"] == ["week_level"]
    assert payload["drift"]["current_days"] == 11


def test_the_panel_is_written_to_the_site_folder(tmp_path, demand):
    data_dir, models_dir, site_dir = tmp_path / "data", tmp_path / "models", tmp_path / "site"
    table = pd.DataFrame(
        {
            "timestamp": demand.index,
            "value": demand.to_numpy(),
            "anomaly": np.zeros(len(demand), bool),
        }
    )
    store.write_table(table, store.processed_dir(config.DEMAND.name, data_dir), "timestamp")
    models_dir.mkdir()
    registry = {"current": ENTRY["version"], "models": [ENTRY]}
    (models_dir / "registry.json").write_text(json.dumps(registry), encoding="utf-8")
    log_days(data_dir, "2026-11-15", 3)

    path = write_panel(data_dir, models_dir, site_dir, now=NOW)

    assert path == site_dir / "index.html"
    page = path.read_text(encoding="utf-8")
    assert "2026-11-17T23:00:00-05:00" in page
    assert '"drift":null' in page
