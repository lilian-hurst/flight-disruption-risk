"""Build the historical training dataset from real weather data.

Methodology (documented transparently, see README "Limites et méthodologie") :
OpenSky's historical flight endpoints require a registered account and are not
accessible anonymously (confirmed: HTTP 403 "You cannot access historical
flights"). Without a source of real historical delay-per-flight records, this
project uses a documented, rule-based proxy label instead of literal delay
minutes: a given (airport, hour) is labelled "disruption risk" = 1 when the
real historical weather at that airport/hour crosses thresholds that are
well-established in aviation operations as correlating with ATC delays and
ground stops (heavy precipitation, strong wind/gusts, very low cloud ceiling
proxy via cloud cover). This is a weak-supervision technique: the *inputs*
(weather) are 100% real Open-Meteo historical data; only the *label* is a
documented heuristic, not historical airline delay records.

Swap-in path to real delay labels: register a free OpenSky account (adds
access to /flights/*), or use a DOT/Eurocontrol punctuality dataset, and
replace `label_disruption_risk()` with a join against real delay minutes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.airports import AIRPORTS
from src.weather import fetch_historical_weather

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Thresholds documented in README: precipitation, wind gusts, cloud cover
PRECIP_THRESHOLD_MM = 2.0
GUST_THRESHOLD_KMH = 50.0
CLOUDCOVER_THRESHOLD_PCT = 90.0

# Forecast horizon: predict disruption risk this many hours ahead of the
# current observation, using only the current hour's features. This is what
# turns the problem into genuine forecasting instead of restating the rule.
HORIZON_HOURS = 3


def label_disruption_risk(row: pd.Series) -> int:
    return int(
        row["precipitation"] > PRECIP_THRESHOLD_MM
        or row["windgusts_10m"] > GUST_THRESHOLD_KMH
        or row["cloudcover"] > CLOUDCOVER_THRESHOLD_PCT
    )


def build(start_date: str, end_date: str) -> pd.DataFrame:
    frames = []
    for icao, info in AIRPORTS.items():
        print(f"  Récupération météo historique {icao} ({info['name']})...")
        payload = fetch_historical_weather(info["lat"], info["lon"], start_date, end_date)
        hourly = payload["hourly"]
        df = pd.DataFrame(hourly)
        df["airport"] = icao
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data["time"] = pd.to_datetime(data["time"])
    data["hour"] = data["time"].dt.hour
    data["month"] = data["time"].dt.month
    data["dayofweek"] = data["time"].dt.dayofweek
    data = data.sort_values(["airport", "time"]).reset_index(drop=True)
    data["disruption_risk_now"] = data.apply(label_disruption_risk, axis=1)

    # Real forecasting target: will conditions be at risk in HORIZON_HOURS from now?
    # Using the current hour's weather to predict a FUTURE hour's risk avoids the
    # trivial data leakage of predicting disruption_risk_now from the exact same
    # columns it was deterministically derived from (which gave a meaningless 100%
    # accuracy in an earlier version of this script -- see README).
    data["disruption_risk_future"] = (
        data.groupby("airport")["disruption_risk_now"].shift(-HORIZON_HOURS)
    )
    data = data.dropna(subset=["disruption_risk_future"]).reset_index(drop=True)
    data["disruption_risk_future"] = data["disruption_risk_future"].astype(int)
    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-06-01")
    parser.add_argument("--end", default="2026-09-17")
    parser.add_argument("--out", default=str(DATA_DIR / "training_data.csv"))
    args = parser.parse_args()

    df = build(args.start, args.end)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(args.out, index=False)
    print(
        f"Dataset écrit : {args.out} ({len(df)} lignes, "
        f"{df['disruption_risk_future'].mean():.1%} en risque à +{HORIZON_HOURS}h)"
    )
