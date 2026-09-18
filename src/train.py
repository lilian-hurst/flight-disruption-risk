"""Train an XGBoost classifier on the historical weather-risk dataset.

Tracks the training's carbon footprint with CodeCarbon (same pattern as the
"Prédiction d'Alertes Foudre" project), and reports standard classification
metrics on a held-out test split.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "training_data.csv"
MODEL_PATH = ROOT / "models" / "model.json"
ENCODER_PATH = ROOT / "models" / "airport_encoder.json"
METRICS_PATH = ROOT / "models" / "metrics.json"

FEATURES = [
    "temperature_2m", "windspeed_10m", "windgusts_10m", "precipitation",
    "cloudcover", "hour", "month", "dayofweek", "airport_code",
]
# Predict the FUTURE risk label (+3h) from CURRENT weather features -- see
# build_dataset.py for why this avoids trivial same-hour data leakage.
TARGET = "disruption_risk_future"


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} introuvable. Lance d'abord : python3 src/build_dataset.py"
        )
    return pd.read_csv(DATA_PATH)


def main():
    df = load_dataset()

    encoder = LabelEncoder()
    df["airport_code"] = encoder.fit_transform(df["airport"])

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    try:
        from codecarbon import EmissionsTracker
        tracker = EmissionsTracker(
            project_name="flight-delay-predictor", output_dir=str(ROOT / "models"),
            log_level="error", save_to_file=True,
        )
        tracker.start()
        tracking = True
    except Exception as e:  # pragma: no cover - CodeCarbon is best-effort
        print(f"(CodeCarbon indisponible, entraînement sans suivi carbone : {e})")
        tracking = False

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        eval_metric="logloss", random_state=42,
    )
    model.fit(X_train, y_train)

    emissions_kg = None
    if tracking:
        emissions_kg = tracker.stop()

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "positive_rate": float(y.mean()),
        "emissions_kg_co2eq": emissions_kg,
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save_model(str(MODEL_PATH))
    ENCODER_PATH.write_text(json.dumps({
        "classes": encoder.classes_.tolist()
    }, indent=2))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print("Modèle entraîné.")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"Modèle sauvegardé -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
