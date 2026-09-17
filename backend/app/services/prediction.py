from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from backend.app.schemas import PredictionRequest, PredictionResponse
from ml_system.persistence import load_model


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def load_prediction_model() -> Pipeline:
    """Load the persisted churn prediction model."""
    return load_model(MODEL_PATH)


def predict_churn(
    request: PredictionRequest,
    model: Pipeline | None = None,
) -> PredictionResponse:
    """Generate a churn prediction for one customer."""

    prediction_model = model or load_prediction_model()

    input_data = pd.DataFrame(
        [
            {
                "gender": request.gender,
                "SeniorCitizen": request.SeniorCitizen,
                "Partner": request.Partner,
                "Dependents": request.Dependents,
                "tenure": request.tenure,
                "PhoneService": request.PhoneService,
                "MultipleLines": request.MultipleLines,
                "InternetService": request.InternetService,
                "OnlineSecurity": request.OnlineSecurity,
                "OnlineBackup": request.OnlineBackup,
                "DeviceProtection": request.DeviceProtection,
                "TechSupport": request.TechSupport,
                "StreamingTV": request.StreamingTV,
                "StreamingMovies": request.StreamingMovies,
                "Contract": request.Contract,
                "PaperlessBilling": request.PaperlessBilling,
                "PaymentMethod": request.PaymentMethod,
                "MonthlyCharges": request.MonthlyCharges,
                "TotalCharges": request.TotalCharges,
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    prediction = prediction_model.predict(input_data)[0]
    probabilities = prediction_model.predict_proba(input_data)[0]

    return PredictionResponse(
        prediction=str(prediction),
        churn_probability=float(probabilities[1]),
    )