"""Minimal client for Garmin's official Connect Personal API.

Authentication is a long-lived personal API key (created at
https://connect.garmin.com/app/settings/personalApiToken) that is exchanged
for a 24-hour OAuth2 bearer token. There are no refresh tokens: when the token
is about to expire we simply exchange the key again. Everything is kept in
memory so nothing secret is ever written to the (ephemeral) host disk.

The API is read-only ("pull only"). It does not expose training status,
race predictions, lactate threshold or personal records — those tools still
need the legacy garminconnect login (see server.py).
"""
import base64
import io
import json
import os
import threading
import zipfile
from collections.abc import Mapping
from datetime import date
from time import time
from typing import Any

import requests

API_KEY_ENV_VAR = "GARMIN_PERSONAL_API_KEY"
BASE_URL_ENV_VAR = "GARMIN_PERSONAL_API_BASE_URL"
DEFAULT_BASE_URL = "https://personal.connectapi.garmin.com"
TOKEN_PATH = "/personalapitoken-service/personal-api/personal-api-key/token"
TIMEOUT_SECONDS = 30.0
# Exchange the key again when the cached token has less than this left.
_REFRESH_WINDOW_SECONDS = 60 * 60
# Used only if the token's `exp` claim can't be read (should not happen).
_FALLBACK_TOKEN_LIFETIME_SECONDS = 23 * 60 * 60

KEY_PAGE_URL = "https://connect.garmin.com/app/settings/personalApiToken"


class PersonalApiError(Exception):
    """Raised for any Personal API failure. Never contains the API key."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def api_key_from_env() -> str | None:
    key = os.environ.get(API_KEY_ENV_VAR, "").strip()
    return key or None


def _iso(value: str | date) -> str:
    return value.isoformat() if isinstance(value, date) else value


def _jwt_expiration(token: str) -> float:
    try:
        payload = token.split(".")[1]
        decoded = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        return float(decoded["exp"])
    except (IndexError, KeyError, TypeError, ValueError):
        return time() + _FALLBACK_TOKEN_LIFETIME_SECONDS


class PersonalApi:
    """Authenticated, read-only access to one Garmin account via its API key."""

    def __init__(self, api_key: str, session: requests.Session | None = None) -> None:
        if not api_key or not api_key.strip():
            raise PersonalApiError(f"{API_KEY_ENV_VAR} is empty")
        self._api_key = api_key.strip()
        self._session = session or requests.Session()
        self._base_url = os.environ.get(BASE_URL_ENV_VAR, DEFAULT_BASE_URL).rstrip("/")
        self._token: str | None = None
        self._expires_at = 0.0
        self._lock = threading.Lock()

    def __repr__(self) -> str:  # never leak the key through logs/tracebacks
        return f"{type(self).__name__}(base_url={self._base_url!r})"

    # -- auth ---------------------------------------------------------------

    def _bearer(self, force: bool = False) -> str:
        with self._lock:
            if not force and self._token and self._expires_at - time() > _REFRESH_WINDOW_SECONDS:
                return self._token
            response = self._session.post(
                f"{self._base_url}{TOKEN_PATH}",
                headers={"Accept": "*/*", "Authorization": f"ApiKey {self._api_key}"},
                timeout=TIMEOUT_SECONDS,
            )
            if not response.ok:
                raise PersonalApiError(
                    f"Garmin rejected the Personal API key (HTTP {response.status_code}). "
                    f"It may have expired or been deleted — create a new one at {KEY_PAGE_URL} "
                    f"and update {API_KEY_ENV_VAR}.",
                    response.status_code,
                )
            token = response.text.strip()
            if not token:
                raise PersonalApiError("Garmin returned an empty access token")
            self._token = token
            self._expires_at = _jwt_expiration(token)
            return token

    # -- http ---------------------------------------------------------------

    def _request(self, path: str, params: Mapping[str, Any] | None, accept: str) -> requests.Response:
        url = f"{self._base_url}{path}"
        for attempt in (1, 2):
            token = self._bearer(force=attempt == 2)
            response = self._session.get(
                url,
                headers={"Accept": accept, "Authorization": f"Bearer {token}"},
                params=params,
                timeout=TIMEOUT_SECONDS,
            )
            # A token can be revoked before `exp`; one forced re-exchange covers it.
            if response.status_code == 401 and attempt == 1:
                continue
            break
        if not response.ok:
            raise PersonalApiError(f"Garmin request failed: HTTP {response.status_code} for {path}", response.status_code)
        return response

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        response = self._request(path, params, "application/json")
        return response.json() if response.content else None

    # -- activities ---------------------------------------------------------

    def search_activities(
        self,
        *,
        start: int = 0,
        limit: int = 20,
        start_date: str | date | None = None,
        end_date: str | date | None = None,
    ) -> Any:
        params: dict[str, Any] = {"start": start, "limit": limit}
        if start_date is not None:
            params["startDate"] = _iso(start_date)
        if end_date is not None:
            params["endDate"] = _iso(end_date)
        return self.get("/activitylist-service/activities/search/activities", params)

    def get_activity(self, activity_id: int | str) -> Any:
        return self.get(f"/activity-service/activity/{_activity_id(activity_id)}")

    def download_fit(self, activity_id: int | str) -> bytes:
        """Return the raw .fit bytes of an activity (the API serves them zipped)."""
        response = self._request(
            f"/download-service/files/activity/{_activity_id(activity_id)}",
            None,
            "application/zip, application/x-zip-compressed, application/octet-stream",
        )
        content = response.content
        if not content:
            raise PersonalApiError("Garmin returned an empty activity file")
        if content[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                names = [n for n in archive.namelist() if n.lower().endswith(".fit")]
                if not names:
                    raise PersonalApiError("The activity archive did not contain a .fit file")
                return archive.read(names[0])
        return content

    # -- daily / wellness ---------------------------------------------------

    def get_daily_summaries(self, start_date: str | date, end_date: str | date) -> Any:
        return self.get(
            "/usersummary-service/v2/dailysummaries",
            {"startDate": _iso(start_date), "endDate": _iso(end_date)},
        )

    def get_daily_sleep(self, calendar_date: str | date) -> Any:
        return self.get("/sleep-service/sleep/dailySleep", {"date": _iso(calendar_date)})

    def get_daily_heart_rate(self, calendar_date: str | date) -> Any:
        return self.get("/wellness-service/wellness/dailyHeartRate", {"date": _iso(calendar_date)})

    def get_daily_body_battery(self, start_date: str | date, end_date: str | date) -> Any:
        return self.get(
            "/wellness-service/wellness/bodyBattery/reports/daily",
            {"startDate": _iso(start_date), "endDate": _iso(end_date)},
        )

    def get_daily_hrv(self, calendar_date: str | date) -> Any:
        return self.get(f"/hrv-service/hrv/{_iso(calendar_date)}")

    def get_daily_vo2_max(self, start_date: str | date, end_date: str | date) -> Any:
        return self.get(f"/metrics-service/metrics/maxmet/daily/{_iso(start_date)}/{_iso(end_date)}")


def _activity_id(activity_id: int | str) -> str:
    text = str(activity_id).strip()
    if not (text.isascii() and text.isdecimal() and int(text) > 0):
        raise PersonalApiError("activity_id must be a positive integer")
    return text


# -- FIT parsing -----------------------------------------------------------


def laps_from_fit(fit_bytes: bytes) -> list[dict]:
    """Extract laps from a .fit file, shaped like Garmin Connect's `lapDTOs`.

    Keys match what server._fmt_lap reads, so the Personal API path returns
    exactly the same lap schema as the legacy path.
    """
    import fitdecode  # imported lazily: only needed for the laps tool

    raw_laps: list[dict] = []
    sport: str | None = None
    with fitdecode.FitReader(io.BytesIO(fit_bytes)) as reader:
        for frame in reader:
            if not isinstance(frame, fitdecode.FitDataMessage):
                continue
            if frame.name == "session" and sport is None:
                sport = frame.get_value("sport", fallback=None)
            elif frame.name == "lap":
                raw_laps.append({f.name: f.value for f in frame.fields})

    # FIT stores running cadence per leg; Garmin Connect reports steps/min.
    cadence_factor = 2 if sport == "running" else 1
    laps = []
    for index, lap in enumerate(raw_laps, start=1):
        cadence = lap.get("avg_running_cadence", lap.get("avg_cadence"))
        elapsed = lap.get("total_elapsed_time")
        timer = lap.get("total_timer_time")
        intensity = lap.get("intensity")
        laps.append(
            {
                "lapIndex": index,
                "distance": lap.get("total_distance"),
                "duration": timer,
                "elapsedDuration": elapsed,
                "movingDuration": lap.get("total_moving_time", timer),
                "averageHR": lap.get("avg_heart_rate"),
                "maxHR": lap.get("max_heart_rate"),
                "averagePower": lap.get("avg_power"),
                "maxPower": lap.get("max_power"),
                "averageRunCadence": cadence * cadence_factor if isinstance(cadence, (int, float)) else None,
                "intensityType": str(intensity).upper() if intensity is not None else None,
                "elevationGain": lap.get("total_ascent"),
                "elevationLoss": lap.get("total_descent"),
            }
        )
    return laps
