"""Thin client for the hourly endpoint of XM's public REST API."""

import time
from collections.abc import Callable
from datetime import date, timedelta

import pandas as pd
import requests

from demand_forecast import config
from demand_forecast.data.schema import DATE_DTYPE, HOUR_COLUMNS, WIDE_COLUMNS

_API_HOUR_KEYS = [f"Hour{hour:02d}" for hour in range(1, 25)]
_NO_DATA_MARKER = "No existen datos"


class XMError(Exception):
    """Base class for every failure of the XM API."""


class XMRequestError(XMError):
    """XM rejected the request. It is a usage error: retrying will not help."""


class XMUnavailableError(XMError):
    """XM did not answer after every retry."""


class XMResponseError(XMError):
    """XM answered with a payload this client does not understand."""


def split_windows(
    start: date, end: date, max_days: int = config.XM_MAX_WINDOW_DAYS
) -> list[tuple[date, date]]:
    """Split [start, end] into consecutive windows of at most `max_days` calendar days."""
    if start > end:
        raise ValueError(f"start ({start}) is after end ({end})")
    windows = []
    cursor = start
    while cursor <= end:
        stop = min(cursor + timedelta(days=max_days - 1), end)
        windows.append((cursor, stop))
        cursor = stop + timedelta(days=1)
    return windows


def empty_wide() -> pd.DataFrame:
    """Return the wide table with no rows."""
    frame = pd.DataFrame({column: pd.Series(dtype=float) for column in HOUR_COLUMNS})
    frame.insert(0, "date", pd.Series(dtype=DATE_DTYPE))
    return frame


class XMClient:
    """Download hourly series, one request at a time."""

    def __init__(
        self,
        session: requests.Session | None = None,
        *,
        url: str = config.XM_HOURLY_URL,
        timeout: float = config.XM_TIMEOUT_SECONDS,
        max_attempts: int = config.XM_MAX_ATTEMPTS,
        backoff_seconds: float = config.XM_BACKOFF_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._session = session or requests.Session()
        self._url = url
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds
        self._sleep = sleep

    def fetch_hourly(self, metric_id: str, entity: str, start: date, end: date) -> pd.DataFrame:
        """Return one row per day in [start, end] with columns `date` and `hour_01`..`hour_24`.

        Days XM has no data for are simply absent. Dates before the start of XM's history are
        skipped, because a window that starts there fails even when it contains valid days.
        """
        start = max(start, config.XM_HISTORY_START)
        if start > end:
            return empty_wide()
        windows = (
            self._fetch_window(metric_id, entity, window_start, window_end)
            for window_start, window_end in split_windows(start, end)
        )
        frames = [frame for frame in windows if not frame.empty]
        if not frames:
            return empty_wide()
        frame = pd.concat(frames, ignore_index=True).sort_values("date", ignore_index=True)
        if frame["date"].duplicated().any():
            raise XMResponseError(f"XM returned repeated dates for {metric_id}/{entity}")
        return frame

    def _fetch_window(self, metric_id: str, entity: str, start: date, end: date) -> pd.DataFrame:
        request = f"{metric_id}/{entity} {start}..{end}"
        body = {
            "MetricId": metric_id,
            "StartDate": start.isoformat(),
            "EndDate": end.isoformat(),
            "Entity": entity,
            "Filter": [],
        }
        response = self._post(body, request)
        if response.status_code == 400 and _NO_DATA_MARKER in response.text:
            return empty_wide()
        if response.status_code != 200:
            raise XMRequestError(
                f"XM rejected {request}: HTTP {response.status_code}: {response.text[:200]}"
            )
        try:
            payload = response.json()
        except ValueError as error:
            raise XMResponseError(f"XM did not return JSON for {request}") from error
        return _parse(payload, request)

    def _post(self, body: dict, request: str) -> requests.Response:
        failure = ""
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._session.post(self._url, json=body, timeout=self._timeout)
            except (requests.ConnectionError, requests.Timeout) as error:
                failure = repr(error)
            else:
                if response.status_code != 429 and response.status_code < 500:
                    return response
                failure = f"HTTP {response.status_code}"
            if attempt < self._max_attempts:
                self._sleep(self._backoff_seconds * 2 ** (attempt - 1))
        raise XMUnavailableError(
            f"XM did not answer {request} after {self._max_attempts} attempts: {failure}"
        )


def _parse(payload: object, request: str) -> pd.DataFrame:
    rows = []
    try:
        for item in payload["Items"]:
            entities = item["HourlyEntities"]
            if len(entities) != 1:
                raise XMResponseError(
                    f"expected one entity per day for {request}, got {len(entities)}"
                )
            values = entities[0]["Values"]
            hours = (values.get(key) for key in _API_HOUR_KEYS)
            rows.append([item["Date"], *(None if hour == "" else hour for hour in hours)])
        frame = pd.DataFrame(rows, columns=WIDE_COLUMNS)
        frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d").astype(DATE_DTYPE)
        frame[HOUR_COLUMNS] = frame[HOUR_COLUMNS].apply(pd.to_numeric).astype(float)
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise XMResponseError(f"unexpected payload for {request}: {error!r}") from error
    return frame
