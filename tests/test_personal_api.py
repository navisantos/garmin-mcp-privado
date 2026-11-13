"""Tests for the Garmin Personal API backend.

No network: HTTP is faked, and the FIT file is a tiny hand-built binary so the
lap parser runs against real FIT bytes.
"""
import base64
import io
import json
import struct
import zipfile

import pytest

from garmin_mcp import personal_api, server
from garmin_mcp.personal_api import PersonalApi, PersonalApiError

# --- helpers ----------------------------------------------------------------


def _jwt(exp: float) -> str:
    def enc(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")

    return f"{enc({'alg': 'none'})}.{enc({'exp': exp})}.sig"


class FakeResponse:
    def __init__(self, status=200, text="", payload=None, content=None):
        self.status_code = status
        self.ok = status < 400
        self.text = text
        self._payload = payload
        self.content = content if content is not None else (json.dumps(payload).encode() if payload is not None else b"x")

    def json(self):
        return self._payload


class FakeSession:
    """Records calls; `get_responses` is consumed in order."""

    def __init__(self, tokens, get_responses):
        self.tokens = list(tokens)
        self.get_responses = list(get_responses)
        self.posts = []
        self.gets = []

    def post(self, url, headers=None, timeout=None):
        self.posts.append((url, headers))
        return FakeResponse(text=self.tokens.pop(0))

    def get(self, url, headers=None, params=None, timeout=None):
        self.gets.append((url, headers, params))
        return self.get_responses.pop(0)


_FIT_CRC_TABLE = [0x0000, 0xCC01, 0xD801, 0x1400, 0xF001, 0x3C00, 0x2800, 0xE401,
                  0xA001, 0x6C00, 0x7800, 0xB401, 0x5000, 0x9C01, 0x8801, 0x4400]


def _crc16(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc = ((crc >> 4) & 0x0FFF) ^ _FIT_CRC_TABLE[crc & 0xF] ^ _FIT_CRC_TABLE[byte & 0xF]
        crc = ((crc >> 4) & 0x0FFF) ^ _FIT_CRC_TABLE[crc & 0xF] ^ _FIT_CRC_TABLE[(byte >> 4) & 0xF]
    return crc


def _definition(local: int, global_num: int, fields: list[tuple[int, int, int]]) -> bytes:
    out = bytes([0x40 | local, 0, 0]) + struct.pack("<H", global_num) + bytes([len(fields)])
    for num, size, base in fields:
        out += bytes([num, size, base])
    return out


def _build_fit(laps: list[dict], sport: int = 1) -> bytes:
    """A minimal FIT file: one session (sport) + N laps."""
    body = _definition(0, 18, [(5, 1, 0x00)]) + bytes([0x00, sport])  # session.sport (enum)
    body += _definition(1, 19, [
        (7, 4, 0x86),   # total_elapsed_time (ms)
        (8, 4, 0x86),   # total_timer_time (ms)
        (9, 4, 0x86),   # total_distance (cm)
        (15, 1, 0x02),  # avg_heart_rate
        (16, 1, 0x02),  # max_heart_rate
        (23, 1, 0x00),  # intensity (enum)
    ])
    for lap in laps:
        body += bytes([0x01]) + struct.pack(
            "<IIIBBB",
            int(lap["elapsed_s"] * 1000), int(lap["timer_s"] * 1000), int(lap["distance_m"] * 100),
            lap["avg_hr"], lap["max_hr"], lap["intensity"],
        )
    header = struct.pack("<BBHI4sH", 14, 0x20, 2000, len(body), b".FIT", 0)
    payload = header + body
    return payload + struct.pack("<H", _crc16(payload))


# --- auth -------------------------------------------------------------------


def test_token_is_exchanged_once_and_reused(monkeypatch):
    session = FakeSession([_jwt(10**10)], [FakeResponse(payload=[]), FakeResponse(payload=[])])
    api = PersonalApi("my-key", session=session)
    api.get("/a")
    api.get("/b")
    assert len(session.posts) == 1
    assert session.posts[0][1]["Authorization"] == "ApiKey my-key"
    assert session.gets[0][1]["Authorization"].startswith("Bearer ")


def test_token_near_expiry_is_refreshed(monkeypatch):
    monkeypatch.setattr(personal_api, "time", lambda: 1000.0)
    # first token expires in 30 min (< 1h window) -> refreshed on the next call
    session = FakeSession([_jwt(1000 + 1800), _jwt(1000 + 86400)], [FakeResponse(payload=1), FakeResponse(payload=2)])
    api = PersonalApi("k", session=session)
    api.get("/a")
    api.get("/b")
    assert len(session.posts) == 2


def test_401_forces_one_token_refresh_then_succeeds():
    session = FakeSession([_jwt(10**10), _jwt(10**10)], [FakeResponse(status=401), FakeResponse(payload={"ok": 1})])
    api = PersonalApi("k", session=session)
    assert api.get("/x") == {"ok": 1}
    assert len(session.posts) == 2


def test_rejected_key_gives_actionable_error_without_leaking_key():
    class Rejecting(FakeSession):
        def post(self, url, headers=None, timeout=None):
            return FakeResponse(status=401, text="nope")

    api = PersonalApi("super-secret-key", session=Rejecting([], []))
    with pytest.raises(PersonalApiError) as exc:
        api.get("/x")
    assert "super-secret-key" not in str(exc.value)
    assert "personalApiToken" in str(exc.value)
    assert "super-secret-key" not in repr(api)


def test_activity_id_is_validated():
    api = PersonalApi("k", session=FakeSession([], []))
    with pytest.raises(PersonalApiError):
        api.get_activity("../../etc/passwd")


# --- FIT --------------------------------------------------------------------


def test_laps_from_fit_parses_real_fit_bytes():
    fit = _build_fit([
        {"elapsed_s": 250.0, "timer_s": 240.0, "distance_m": 1000.0, "avg_hr": 164, "max_hr": 170, "intensity": 0},
        {"elapsed_s": 60.0, "timer_s": 60.0, "distance_m": 150.0, "avg_hr": 120, "max_hr": 130, "intensity": 1},
    ])
    laps = personal_api.laps_from_fit(fit)
    assert [lap["lapIndex"] for lap in laps] == [1, 2]
    assert laps[0]["distance"] == 1000.0
    assert laps[0]["duration"] == 240.0
    assert laps[0]["elapsedDuration"] == 250.0
    assert laps[0]["averageHR"] == 164
    assert laps[0]["intensityType"] == "ACTIVE"
    assert laps[1]["intensityType"] == "REST"


def test_download_fit_unzips_the_archive():
    fit = _build_fit([{"elapsed_s": 1.0, "timer_s": 1.0, "distance_m": 1.0, "avg_hr": 1, "max_hr": 1, "intensity": 0}])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("123_ACTIVITY.fit", fit)
    session = FakeSession([_jwt(10**10)], [FakeResponse(content=buf.getvalue())])
    assert PersonalApi("k", session=session).download_fit("123") == fit


# --- server routing ---------------------------------------------------------


class FakePersonal:
    def search_activities(self, **kwargs):
        return [{
            "activityId": 7, "activityName": "HYROX SP", "activityType": {"typeKey": "hiit"},
            "distance": 8000.0, "duration": 4000.0, "averageHR": 160, "startTimeLocal": "2026-10-18 08:00:00",
        }]

    def get_daily_summaries(self, start, end):
        return {"data": [{"calendarDate": start, "totalSteps": 9000}]}

    def get_activity(self, activity_id):
        return {"activityId": int(activity_id), "summaryDTO": {"duration": 10.0, "movingDuration": 9.0, "elapsedDuration": 12.0}}

    def download_fit(self, activity_id):
        return _build_fit([{"elapsed_s": 250.0, "timer_s": 240.0, "distance_m": 1000.0, "avg_hr": 164, "max_hr": 170, "intensity": 0}])

    def get_daily_hrv(self, date):
        return {"hrvSummary": {"lastNightAvg": 55}, "hrvReadings": [1] * 500}

    def get_daily_vo2_max(self, start, end):
        return [{"generic": {"vo2MaxPreciseValue": 59.24, "vo2MaxValue": 59, "calendarDate": end}}]


@pytest.fixture
def personal_mode(monkeypatch):
    monkeypatch.setenv("GARMIN_PERSONAL_API_KEY", "k")
    monkeypatch.delenv("GARMIN_EMAIL", raising=False)
    monkeypatch.delenv("GARMIN_PASSWORD", raising=False)
    monkeypatch.setenv("GARMIN_TOKENS", "/nonexistent/garmin-tokens")
    monkeypatch.setattr(server, "get_personal", lambda: FakePersonal())
    # any accidental legacy call would blow up loudly
    monkeypatch.setattr(server, "get_client", lambda: (_ for _ in ()).throw(AssertionError("legacy used")))


def test_list_recent_activities_uses_personal_api(personal_mode):
    out = json.loads(server.list_recent_activities(limit=1))
    assert out[0]["id"] == 7 and out[0]["distance_km"] == 8.0


def test_daily_stats_unwraps_data_array(personal_mode):
    assert json.loads(server.get_daily_stats("2026-11-13"))["totalSteps"] == 9000


def test_activity_details_includes_timing(personal_mode):
    out = json.loads(server.get_activity_details("7"))
    assert out["timing"]["stopped_time_s"] == 3.0


def test_activity_laps_come_from_fit(personal_mode):
    out = json.loads(server.get_activity_laps("7"))
    assert out["lap_count"] == 1
    assert out["laps"][0]["distance_m"] == 1000.0
    assert out["laps"][0]["intensity"] == "ACTIVE"


def test_hrv_strips_readings(personal_mode):
    out = json.loads(server.get_hrv("2026-11-13"))
    assert "hrvReadings" not in out and out["hrvSummary"]["lastNightAvg"] == 55


def test_unsupported_tool_returns_clear_error_without_legacy(personal_mode):
    out = json.loads(server.get_training_status("2026-11-13"))
    assert "Personal API" in out["error"]


def test_performance_metrics_mixes_backends(personal_mode):
    out = json.loads(server.get_performance_metrics("2026-11-13"))
    assert out["vo2max"]["value"] == 59.2
    assert "Personal API" in out["lactate_threshold"]["error"]
    assert "Personal API" in out["race_predictions"]["error"]


def test_legacy_path_unchanged_when_no_key(monkeypatch):
    monkeypatch.delenv("GARMIN_PERSONAL_API_KEY", raising=False)
    assert server._personal_enabled() is False
