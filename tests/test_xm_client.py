import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
import requests

from demand_forecast import config
from demand_forecast.data.schema import DATE_DTYPE, HOUR_COLUMNS, WIDE_COLUMNS
from demand_forecast.data.xm_client import (
    XMClient,
    XMRequestError,
    XMResponseError,
    XMUnavailableError,
    split_windows,
)

FIXTURES = Path(__file__).parent / "fixtures"
NO_DATA = "No existen datos para el rango de fechas consultado"
RANGE_TOO_LONG = "Rango de consulta supera el maximo permitido"


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: object = None, text: str = "") -> None:
        self.status_code = status_code
        self.text = text
        self._payload = payload

    def json(self) -> object:
        if self._payload is None:
            raise ValueError("not JSON")
        return self._payload


class FakeSession:
    """Answer each POST with the next scripted reply, or with `handler(body)`."""

    def __init__(self, replies: list | None = None, handler=None) -> None:
        self.replies = list(replies or [])
        self.handler = handler
        self.bodies: list[dict] = []

    def post(self, url: str, json: dict, timeout: float) -> FakeResponse:
        self.bodies.append(json)
        reply = self.handler(json) if self.handler else self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def payload(days: dict[str, list]) -> dict:
    return {
        "Items": [
            {
                "Date": day,
                "HourlyEntities": [
                    {
                        "Id": "Sistema",
                        "Values": {
                            "code": "Sistema",
                            **{f"Hour{i:02d}": v for i, v in enumerate(hours, 1)},
                        },
                    }
                ],
            }
            for day, hours in days.items()
        ]
    }


def ok(days: dict[str, list]) -> FakeResponse:
    return FakeResponse(200, payload(days))


def client(session: FakeSession, **kwargs) -> XMClient:
    return XMClient(session, sleep=lambda seconds: None, **kwargs)


HOURS = [f"{1000 + i}.50000" for i in range(24)]


class TestSplitWindows:
    def test_windows_cover_the_range_without_overlap(self):
        start, end = date(2026, 1, 1), date(2026, 12, 31)
        windows = split_windows(start, end)
        assert windows[0][0] == start
        assert windows[-1][1] == end
        for (_, previous_end), (next_start, _) in zip(windows, windows[1:], strict=False):
            assert next_start == previous_end + timedelta(days=1)

    def test_no_window_exceeds_the_api_limit(self):
        windows = split_windows(date(2000, 1, 1), date(2026, 11, 11))
        assert max((end - start).days for start, end in windows) == config.XM_MAX_WINDOW_DAYS - 1
        assert all((end - start).days <= 31 for start, end in windows)

    def test_single_day(self):
        day = date(2026, 11, 5)
        assert split_windows(day, day) == [(day, day)]

    def test_start_after_end_is_an_error(self):
        with pytest.raises(ValueError, match="after end"):
            split_windows(date(2026, 1, 2), date(2026, 1, 1))


class TestFetchHourly:
    def test_parses_a_real_payload(self):
        real = json.loads((FIXTURES / "xm_hourly_response.json").read_text(encoding="utf-8"))
        session = FakeSession([FakeResponse(200, real)])

        frame = client(session).fetch_hourly(
            "DemaReal", "Sistema", date(2026, 10, 20), date(2026, 10, 21)
        )

        assert list(frame.columns) == WIDE_COLUMNS
        assert frame["date"].dtype == DATE_DTYPE
        assert list(frame["date"]) == [pd.Timestamp("2026-10-20"), pd.Timestamp("2026-10-21")]
        assert all(frame[column].dtype == float for column in HOUR_COLUMNS)
        assert frame.loc[0, "hour_01"] == 9233757.32
        assert frame.loc[0, "hour_24"] == 9574227.10

    def test_sends_the_documented_body(self):
        session = FakeSession([ok({"2026-11-01": HOURS})])
        client(session).fetch_hourly("DemaCome", "Sistema", date(2026, 11, 1), date(2026, 11, 1))
        assert session.bodies == [
            {
                "MetricId": "DemaCome",
                "StartDate": "2026-11-01",
                "EndDate": "2026-11-01",
                "Entity": "Sistema",
                "Filter": [],
            }
        ]

    def test_long_ranges_are_requested_window_by_window_in_order(self):
        def handler(body: dict) -> FakeResponse:
            days = pd.date_range(body["StartDate"], body["EndDate"], freq="D")
            return ok({day.strftime("%Y-%m-%d"): HOURS for day in days})

        session = FakeSession(handler=handler)
        frame = client(session).fetch_hourly(
            "DemaCome", "Sistema", date(2026, 1, 1), date(2026, 3, 31)
        )

        starts = [body["StartDate"] for body in session.bodies]
        assert starts == sorted(starts)
        assert len(session.bodies) == 3
        assert len(frame) == 90
        assert frame["date"].is_monotonic_increasing
        assert frame["date"].is_unique

    def test_never_asks_for_dates_before_the_start_of_history(self):
        session = FakeSession(handler=lambda body: FakeResponse(400, text=NO_DATA))
        client(session).fetch_hourly("DemaCome", "Sistema", date(1999, 12, 20), date(2000, 1, 10))
        assert session.bodies[0]["StartDate"] == "2000-01-01"

    def test_range_entirely_before_history_makes_no_request(self):
        session = FakeSession()
        frame = client(session).fetch_hourly(
            "DemaCome", "Sistema", date(1995, 1, 1), date(1995, 1, 31)
        )
        assert frame.empty
        assert list(frame.columns) == WIDE_COLUMNS
        assert session.bodies == []

    def test_no_data_answer_is_an_empty_result_not_an_error(self):
        session = FakeSession([FakeResponse(400, text=NO_DATA)])
        frame = client(session).fetch_hourly(
            "DemaCome", "Sistema", date(2026, 11, 10), date(2026, 11, 11)
        )
        assert frame.empty
        assert list(frame.columns) == WIDE_COLUMNS
        assert frame["date"].dtype == DATE_DTYPE

    def test_windows_without_data_are_skipped(self):
        session = FakeSession([FakeResponse(400, text=NO_DATA), ok({"2026-02-10": HOURS})])
        frame = client(session).fetch_hourly(
            "DemaCome", "Sistema", date(2026, 1, 1), date(2026, 2, 28)
        )
        assert list(frame["date"]) == [pd.Timestamp("2026-02-10")]

    def test_range_too_long_is_a_usage_error(self):
        session = FakeSession([FakeResponse(400, text=RANGE_TOO_LONG)])
        with pytest.raises(XMRequestError, match="Rango de consulta"):
            client(session).fetch_hourly("DemaCome", "Sistema", date(2026, 1, 1), date(2026, 1, 2))
        assert len(session.bodies) == 1

    def test_missing_and_empty_hours_become_nan(self):
        hours = [*HOURS[:22], "", None]
        session = FakeSession([ok({"2026-11-07": hours})])
        frame = client(session).fetch_hourly(
            "DemaCome", "Sistema", date(2026, 11, 7), date(2026, 11, 7)
        )
        assert frame.loc[0, "hour_22"] == 1021.5
        assert frame.loc[0, ["hour_23", "hour_24"]].isna().all()
        assert frame["hour_24"].dtype == float

    @pytest.mark.parametrize(
        "bad",
        [
            FakeResponse(200, payload=None),
            FakeResponse(200, payload={"Unexpected": []}),
            FakeResponse(200, payload={"Items": [{"Date": "2026-11-01"}]}),
            FakeResponse(200, payload=payload({"2026-11-01": ["abc", *HOURS[1:]]})),
            FakeResponse(200, payload=payload({"01/11/2026": HOURS})),
        ],
        ids=["not-json", "no-items", "no-entities", "non-numeric", "bad-date"],
    )
    def test_unexpected_payloads_raise_a_clear_error(self, bad):
        with pytest.raises(XMResponseError):
            client(FakeSession([bad])).fetch_hourly(
                "DemaCome", "Sistema", date(2026, 11, 1), date(2026, 11, 1)
            )

    def test_several_entities_per_day_are_rejected(self):
        body = payload({"2026-11-01": HOURS})
        body["Items"][0]["HourlyEntities"] *= 2
        with pytest.raises(XMResponseError, match="one entity per day"):
            client(FakeSession([FakeResponse(200, body)])).fetch_hourly(
                "DemaCome", "Agente", date(2026, 11, 1), date(2026, 11, 1)
            )


class TestRetries:
    DAY = date(2026, 11, 1)

    def fetch(self, session: FakeSession, **kwargs) -> pd.DataFrame:
        return client(session, **kwargs).fetch_hourly("DemaCome", "Sistema", self.DAY, self.DAY)

    @pytest.mark.parametrize(
        "failure",
        [
            FakeResponse(500, text="boom"),
            FakeResponse(503),
            FakeResponse(429),
            requests.ConnectionError("reset"),
            requests.Timeout("slow"),
        ],
        ids=["500", "503", "429", "connection", "timeout"],
    )
    def test_transient_failures_are_retried(self, failure):
        session = FakeSession([failure, failure, ok({"2026-11-01": HOURS})])
        assert len(self.fetch(session)) == 1
        assert len(session.bodies) == 3

    def test_gives_up_after_max_attempts(self):
        session = FakeSession([FakeResponse(500)] * 3)
        with pytest.raises(XMUnavailableError, match="after 3 attempts: HTTP 500"):
            self.fetch(session, max_attempts=3)
        assert len(session.bodies) == 3

    def test_backoff_grows_between_attempts(self):
        waits: list[float] = []
        session = FakeSession([FakeResponse(500)] * 4)
        xm = XMClient(session, sleep=waits.append, max_attempts=4, backoff_seconds=2.0)
        with pytest.raises(XMUnavailableError):
            xm.fetch_hourly("DemaCome", "Sistema", self.DAY, self.DAY)
        assert waits == [2.0, 4.0, 8.0]

    def test_usage_errors_are_not_retried(self):
        session = FakeSession([FakeResponse(404, text="not found")])
        with pytest.raises(XMRequestError, match="HTTP 404"):
            self.fetch(session)
        assert len(session.bodies) == 1
