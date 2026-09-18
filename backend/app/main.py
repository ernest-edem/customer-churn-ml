from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app.schemas import PredictionRequest, PredictionResponse
from backend.app.services.prediction import predict_churn


app = FastAPI(
    title="Customer Churn ML API",
    description="API for customer churn prediction.",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Return the API health status."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Generate a customer churn prediction."""
    try:
        return predict_churn(request)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc