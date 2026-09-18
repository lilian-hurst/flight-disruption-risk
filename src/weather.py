"""Weather data access via the free, keyless Open-Meteo API.

Two endpoints are used:
- Historical archive (archive-api.open-meteo.com) for training data
- Forecast/current (api.open-meteo.com) for live scoring at inference time
"""
from __future__ import annotations

import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = "temperature_2m,windspeed_10m,windgusts_10m,precipitation,cloudcover"


def fetch_historical_weather(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    """Fetch hourly historical weather for a location between two ISO dates (YYYY-MM-DD)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": HOURLY_VARS,
        "timezone": "UTC",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def fetch_current_weather(lat: float, lon: float) -> dict:
    """Fetch current + next-hours forecast weather for a location (used at inference time)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": HOURLY_VARS,
        "current_weather": "true",
        "forecast_days": 1,
        "timezone": "UTC",
    }
    resp = requests.get(FORECAST_URL, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def current_hourly_snapshot(lat: float, lon: float) -> dict:
    """Return a flat dict of the weather features for the current hour, ready for the model."""
    data = fetch_current_weather(lat, lon)
    hourly = data["hourly"]
    # Open-Meteo returns the current UTC hour as one of the entries; take the first one
    # (forecast starts at the current hour when forecast_days=1 and no start override is given).
    idx = 0
    return {
        "temperature_2m": hourly["temperature_2m"][idx],
        "windspeed_10m": hourly["windspeed_10m"][idx],
        "windgusts_10m": hourly["windgusts_10m"][idx],
        "precipitation": hourly["precipitation"][idx],
        "cloudcover": hourly["cloudcover"][idx],
        "time": hourly["time"][idx],
    }
