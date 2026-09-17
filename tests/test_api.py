from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


VALID_PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 70.0,
}


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction_endpoint() -> None:
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "churn_probability" in data
    assert data["prediction"] in {"Yes", "No"}
    assert 0 <= data["churn_probability"] <= 1


def test_prediction_rejects_invalid_categorical_value() -> None:
    payload = {
        **VALID_PAYLOAD,
        "gender": "Unknown",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_prediction_rejects_invalid_numeric_value() -> None:
    payload = {
        **VALID_PAYLOAD,
        "tenure": 101,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422