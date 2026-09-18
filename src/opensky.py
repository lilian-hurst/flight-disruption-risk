"""Live air traffic data via the free, keyless OpenSky Network REST API.

Only the /states/all endpoint is used: it is the one endpoint OpenSky allows
without an account. The /flights/* (historical) endpoints require a registered
account (verified during this project: anonymous calls return HTTP 403
"You cannot access historical flights") -- see README for details on this
limitation and how the project works around it.
"""
from __future__ import annotations

import requests

STATES_URL = "https://opensky-network.org/api/states/all"

# Column order returned by /states/all, per OpenSky's documented schema.
STATE_FIELDS = [
    "icao24", "callsign", "origin_country", "time_position", "last_contact",
    "longitude", "latitude", "baro_altitude", "on_ground", "velocity",
    "true_track", "vertical_rate", "sensors", "geo_altitude", "squawk",
    "spi", "position_source",
]

BOX_DEGREES = 0.6  # ~65km around the airport, enough to capture approach/departure traffic
LOW_ALTITUDE_M = 3000  # aircraft below this are considered in approach/climb-out phase

# Some hosting providers' IP ranges get a slow/dropped connection from OpenSky
# (observed on Render's free tier: consistent connect-timeout, works fine from
# a residential/dev IP -- see README "Limites et méthodologie"). Keep this
# short so a blocked connection fails fast instead of stalling the request.
CONNECT_TIMEOUT_S = 6


def fetch_states_near(lat: float, lon: float, box_degrees: float = BOX_DEGREES) -> list[dict]:
    """Fetch live aircraft states within a bounding box around (lat, lon)."""
    params = {
        "lamin": lat - box_degrees,
        "lamax": lat + box_degrees,
        "lomin": lon - box_degrees,
        "lomax": lon + box_degrees,
    }
    resp = requests.get(STATES_URL, params=params, timeout=CONNECT_TIMEOUT_S)
    resp.raise_for_status()
    payload = resp.json()
    states = payload.get("states") or []
    return [dict(zip(STATE_FIELDS, s)) for s in states]


def congestion_features(lat: float, lon: float) -> dict:
    """Turn a snapshot of live traffic around an airport into model-ready features."""
    states = fetch_states_near(lat, lon)
    n_aircraft = len(states)
    if n_aircraft == 0:
        return {
            "n_aircraft": 0, "n_on_ground": 0, "n_low_altitude": 0,
            "avg_velocity": 0.0, "avg_vertical_rate_abs": 0.0,
        }

    n_on_ground = sum(1 for s in states if s.get("on_ground"))
    n_low_altitude = sum(
        1 for s in states
        if s.get("baro_altitude") is not None and s["baro_altitude"] < LOW_ALTITUDE_M
    )
    velocities = [s["velocity"] for s in states if s.get("velocity") is not None]
    vrates = [abs(s["vertical_rate"]) for s in states if s.get("vertical_rate") is not None]

    return {
        "n_aircraft": n_aircraft,
        "n_on_ground": n_on_ground,
        "n_low_altitude": n_low_altitude,
        "avg_velocity": sum(velocities) / len(velocities) if velocities else 0.0,
        "avg_vertical_rate_abs": sum(vrates) / len(vrates) if vrates else 0.0,
    }
