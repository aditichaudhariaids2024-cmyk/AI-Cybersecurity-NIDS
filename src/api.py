
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ============================================================
# PATH
# ============================================================

MODEL_PATH = Path("models/nids_random_forest.pkl")


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

print("Random Forest model loaded successfully.")


# ============================================================
# GET FEATURE NAMES
# ============================================================

FEATURE_NAMES = list(model.feature_names_in_)

print(f"Number of model features: {len(FEATURE_NAMES)}")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Cybersecurity Threat Detection API",
    description="Network Intrusion Detection System using CIC-IDS2017 and Random Forest",
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class NetworkTraffic(BaseModel):
    features: dict[str, float]


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():
    return {
        "message": "AI Cybersecurity Threat Detection API",
        "status": "running",
        "model": "Random Forest",
        "dataset": "CIC-IDS2017",
        "features_required": len(FEATURE_NAMES)
    }


# ============================================================
# FEATURE INFORMATION
# ============================================================

@app.get("/features")
def get_features():
    return {
        "number_of_features": len(FEATURE_NAMES),
        "features": FEATURE_NAMES
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict_traffic(data: NetworkTraffic):

    # --------------------------------------------------------
    # Check missing features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURE_NAMES
        if feature not in data.features
    ]

    if missing_features:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing features",
                "missing_features": missing_features
            }
        )

    # --------------------------------------------------------
    # Check extra features
    # --------------------------------------------------------

    extra_features = [
        feature
        for feature in data.features
        if feature not in FEATURE_NAMES
    ]

    if extra_features:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Unknown features",
                "extra_features": extra_features
            }
        )

    # --------------------------------------------------------
    # Arrange features in exact training order
    # --------------------------------------------------------

    input_data = {
        feature: data.features[feature]
        for feature in FEATURE_NAMES
    }

    df = pd.DataFrame([input_data])

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = int(model.predict(df)[0])

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probabilities = model.predict_proba(df)[0]

    benign_probability = float(probabilities[0])
    attack_probability = float(probabilities[1])

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if prediction == 1:

        result = "ATTACK"

        message = (
            "Potential network intrusion detected. "
            "Traffic has been classified as malicious."
        )

    else:

        result = "BENIGN"

        message = (
            "Traffic appears normal according to the "
            "trained intrusion detection model."
        )

    return {
        "prediction": result,
        "target": prediction,
        "attack_probability": round(attack_probability, 4),
        "benign_probability": round(benign_probability, 4),
        "message": message
    }