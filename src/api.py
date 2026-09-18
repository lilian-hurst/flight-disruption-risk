"""FastAPI service: live disruption-risk scoring for a set of European airports.

GET /risk/{icao} combines two REAL, live public data sources:
  1. The trained model's +3h weather-disruption-risk forecast (see train.py),
     fed with live current weather from Open-Meteo.
  2. Live air traffic congestion around the airport from OpenSky Network
     (aircraft count, how many are low-altitude/on approach, average speed).

The OpenSky congestion numbers are reported as live operational context
alongside the prediction, not fed into the model itself: there is no
anonymous access to OpenSky's historical data (see README), so there is no
honest way to have trained the model on historical congestion -- only on
historical weather, which Open-Meteo's archive does provide for free.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.airports import AIRPORTS, get_airport
from src.opensky import congestion_features
from src.weather import current_hourly_snapshot

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.json"
ENCODER_PATH = ROOT / "models" / "airport_encoder.json"
STATIC_DIR = ROOT / "static"

app = FastAPI(
    title="Flight Disruption Risk API",
    description=(
        "Prévision à +3h du risque de perturbation météo pour un aéroport, "
        "combinée à la congestion aérienne en direct (OpenSky Network)."
    ),
    version="0.1.0",
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def frontend():
    """Serve the single-page dashboard (see static/index.html)."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "Interface non trouvée, voir /docs pour l'API"}

_model: xgb.XGBClassifier | None = None
_airport_classes: list[str] | None = None


def get_model() -> tuple[xgb.XGBClassifier, list[str]]:
    global _model, _airport_classes
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError("Modèle introuvable. Lance d'abord : python3 src/train.py")
        _model = xgb.XGBClassifier()
        _model.load_model(str(MODEL_PATH))
        _airport_classes = json.loads(ENCODER_PATH.read_text())["classes"]
    return _model, _airport_classes


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/airports")
def list_airports():
    return AIRPORTS


@app.get("/risk/{icao}")
def risk(icao: str):
    icao = icao.upper()
    try:
        airport = get_airport(icao)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    model, airport_classes = get_model()
    if icao not in airport_classes:
        raise HTTPException(
            status_code=400,
            detail=f"L'aéroport {icao} n'a pas été vu à l'entraînement.",
        )

    weather = current_hourly_snapshot(airport["lat"], airport["lon"])
    now = datetime.now(timezone.utc)

    features = [[
        weather["temperature_2m"],
        weather["windspeed_10m"],
        weather["windgusts_10m"],
        weather["precipitation"],
        weather["cloudcover"],
        now.hour,
        now.month,
        now.weekday(),
        airport_classes.index(icao),
    ]]
    risk_proba = float(model.predict_proba(features)[0][1])

    try:
        congestion = congestion_features(airport["lat"], airport["lon"])
        congestion_error = None
    except requests.exceptions.ConnectionError:
        # Observed on some hosting providers (e.g. Render free tier): OpenSky
        # is unreachable from their IP range, even though it works fine from
        # a residential/dev connection. Documented in README.
        congestion = None
        congestion_error = (
            "OpenSky Network est inaccessible depuis ce serveur d'hébergement "
            "(fonctionne en local / Docker — voir README)."
        )
    except Exception as e:  # any other failure (rate-limit, timeout, ...)
        congestion = None
        congestion_error = str(e)

    return {
        "airport": {"icao": icao, **airport},
        "forecast_horizon_hours": 3,
        "disruption_risk_score": round(risk_proba, 3),
        "disruption_risk_level": (
            "élevé" if risk_proba > 0.5 else "modéré" if risk_proba > 0.25 else "faible"
        ),
        "current_weather": weather,
        "live_traffic_congestion": congestion,
        "live_traffic_error": congestion_error,
        "queried_at": now.isoformat(),
    }
