"""Central configuration: every tunable value of the project lives here."""

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class SeriesSpec:
    """An hourly XM series: the local name used for folders, plus how to request it."""

    name: str
    metric_id: str
    entity: str


# --- Problem definition (PLAN.md section 4) ---
TIMEZONE = "America/Bogota"
HORIZON_DAYS = 1
# A forecast issued on day D only sees demand up to D - DATA_LAG_DAYS.
# Provisional: measured on 2026-11-10 and 2026-11-11 (docs/data_notes.md section 3).
DATA_LAG_DAYS = 5

# --- Series ---
DEMAND = SeriesSpec(name="demand", metric_id="DemaCome", entity="Sistema")
GENERATION = SeriesSpec(name="generation", metric_id="Gene", entity="Sistema")
SERIES = (DEMAND, GENERATION)

# --- XM API ---
XM_HOURLY_URL = "https://servapibi.xm.com.co/hourly"
XM_HISTORY_START = date(2000, 1, 1)
# Calendar days per request, both ends included. The API rejects EndDate - StartDate > 31.
XM_MAX_WINDOW_DAYS = 31
XM_TIMEOUT_SECONDS = 60
XM_MAX_ATTEMPTS = 4
XM_BACKOFF_SECONDS = 2.0

# --- Storage ---
DATA_DIR = Path(os.environ.get("DEMAND_FORECAST_DATA_DIR", "data"))

# --- Ingestion ---
# Days re-downloaded on every run: partial days fill in later and complete ones get revised.
REDOWNLOAD_DAYS = 31

# --- Data quality (docs/data_notes.md section 3.1) ---
COMPLETENESS_REFERENCE_DAYS = 28
COMPLETENESS_MIN_RATIO = 0.60
DEMAND_VS_GENERATION_MIN_RATIO = 0.95
ANOMALY_REFERENCE_DAYS = 28
ANOMALY_MAX_DEVIATION = 0.35
# Hourly kWh outside this range fail validation. History spans 0.12e6 (2007 blackout) to 12.2e6.
PLAUSIBLE_KWH_MIN = 5e4
PLAUSIBLE_KWH_MAX = 2.5e7

# --- Evaluation (PLAN.md section 4) ---
BACKTEST_MONTHS = 12
MOVING_AVERAGE_WEEKS = 4
HOLIDAY_COUNTRY = "CO"
