import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.json"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="modèle non entraîné : lance `python3 src/train.py` d'abord",
)


@pytest.fixture
def client():
    from src.api import app
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_frontend_served_at_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Flight Disruption Risk" in resp.text


def test_list_airports(client):
    resp = client.get("/airports")
    assert resp.status_code == 200
    assert "LFMN" in resp.json()


def test_risk_unknown_airport(client):
    resp = client.get("/risk/ZZZZ")
    assert resp.status_code == 404


def test_risk_known_airport_mocked(client, monkeypatch):
    fake_weather = {
        "temperature_2m": 18.0, "windspeed_10m": 10.0, "windgusts_10m": 15.0,
        "precipitation": 0.0, "cloudcover": 20, "time": "2026-09-18T12:00",
    }
    fake_congestion = {
        "n_aircraft": 4, "n_on_ground": 1, "n_low_altitude": 2,
        "avg_velocity": 150.0, "avg_vertical_rate_abs": 3.0,
    }
    monkeypatch.setattr("src.api.current_hourly_snapshot", lambda lat, lon: fake_weather)
    monkeypatch.setattr("src.api.congestion_features", lambda lat, lon: fake_congestion)

    resp = client.get("/risk/LFMN")
    assert resp.status_code == 200
    body = resp.json()
    assert body["airport"]["icao"] == "LFMN"
    assert 0.0 <= body["disruption_risk_score"] <= 1.0
    assert body["disruption_risk_level"] in {"faible", "modéré", "élevé"}
    assert body["current_weather"] == fake_weather
    assert body["live_traffic_congestion"] == fake_congestion


def test_risk_survives_opensky_failure(client, monkeypatch):
    fake_weather = {
        "temperature_2m": 18.0, "windspeed_10m": 10.0, "windgusts_10m": 15.0,
        "precipitation": 0.0, "cloudcover": 20, "time": "2026-09-18T12:00",
    }

    def boom(lat, lon):
        raise RuntimeError("OpenSky indisponible")

    monkeypatch.setattr("src.api.current_hourly_snapshot", lambda lat, lon: fake_weather)
    monkeypatch.setattr("src.api.congestion_features", boom)

    resp = client.get("/risk/LFMN")
    assert resp.status_code == 200
    body = resp.json()
    assert body["live_traffic_congestion"] is None
    assert "OpenSky indisponible" in body["live_traffic_error"]


def test_risk_reports_friendly_message_on_connection_error(client, monkeypatch):
    """Some hosting providers (observed on Render) can't reach OpenSky at all --
    this should degrade to a clear, non-technical message rather than a raw
    Python exception string. See src/api.py and README."""
    import requests

    fake_weather = {
        "temperature_2m": 18.0, "windspeed_10m": 10.0, "windgusts_10m": 15.0,
        "precipitation": 0.0, "cloudcover": 20, "time": "2026-09-18T12:00",
    }

    def boom(lat, lon):
        raise requests.exceptions.ConnectionError("Max retries exceeded...")

    monkeypatch.setattr("src.api.current_hourly_snapshot", lambda lat, lon: fake_weather)
    monkeypatch.setattr("src.api.congestion_features", boom)

    resp = client.get("/risk/LFMN")
    assert resp.status_code == 200
    body = resp.json()
    assert body["live_traffic_congestion"] is None
    assert "inaccessible depuis ce serveur d'hébergement" in body["live_traffic_error"]
