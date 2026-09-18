import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.airports import AIRPORTS, get_airport
from src.build_dataset import (
    CLOUDCOVER_THRESHOLD_PCT, GUST_THRESHOLD_KMH, PRECIP_THRESHOLD_MM,
    label_disruption_risk,
)
from src.opensky import congestion_features


def test_airports_reference_data_is_well_formed():
    assert len(AIRPORTS) >= 5
    for icao, info in AIRPORTS.items():
        assert len(icao) == 4
        assert -90 <= info["lat"] <= 90
        assert -180 <= info["lon"] <= 180


def test_get_airport_unknown_raises():
    with pytest.raises(KeyError):
        get_airport("ZZZZ")


def test_get_airport_case_insensitive():
    assert get_airport("lfmn")["iata"] == "NCE"


@pytest.mark.parametrize(
    "precip, gust, cloud, expected",
    [
        (0.0, 0.0, 0, 0),
        (PRECIP_THRESHOLD_MM + 0.1, 0.0, 0, 1),
        (0.0, GUST_THRESHOLD_KMH + 0.1, 0, 1),
        (0.0, 0.0, CLOUDCOVER_THRESHOLD_PCT + 1, 1),
        (PRECIP_THRESHOLD_MM, GUST_THRESHOLD_KMH, CLOUDCOVER_THRESHOLD_PCT, 0),  # exactly at threshold, not over
    ],
)
def test_label_disruption_risk_thresholds(precip, gust, cloud, expected):
    row = pd.Series({
        "precipitation": precip, "windgusts_10m": gust, "cloudcover": cloud,
    })
    assert label_disruption_risk(row) == expected


def test_congestion_features_shape_from_fake_states(monkeypatch):
    fake_states = [
        {"on_ground": False, "baro_altitude": 1500, "velocity": 120, "vertical_rate": 5},
        {"on_ground": True, "baro_altitude": None, "velocity": None, "vertical_rate": None},
        {"on_ground": False, "baro_altitude": 9000, "velocity": 230, "vertical_rate": -1},
    ]
    monkeypatch.setattr("src.opensky.fetch_states_near", lambda lat, lon, box_degrees=0.6: fake_states)

    feats = congestion_features(43.66, 7.21)
    assert feats["n_aircraft"] == 3
    assert feats["n_on_ground"] == 1
    assert feats["n_low_altitude"] == 1  # only the 1500m one is below 3000m
    assert feats["avg_velocity"] == pytest.approx((120 + 230) / 2)


def test_congestion_features_handles_empty_airspace(monkeypatch):
    monkeypatch.setattr("src.opensky.fetch_states_near", lambda lat, lon, box_degrees=0.6: [])
    feats = congestion_features(43.66, 7.21)
    assert feats["n_aircraft"] == 0
    assert feats["avg_velocity"] == 0.0
