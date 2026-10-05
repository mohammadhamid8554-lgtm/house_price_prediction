from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from ml_house_price_prediction.api.schemas import (
    HealthResponse,
    HouseFeatures,
    PredictionResponse,
)
from ml_house_price_prediction.pipelines.prediction_pipeline import PredictPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.pkl"
PREPROCESSOR_PATH = PROJECT_ROOT / "artifacts" / "preprocessor.pkl"

app = FastAPI(
    title="House Price Prediction API",
    description="Estimate a house price from its features.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_prediction_pipeline() -> PredictPipeline:
    return PredictPipeline(MODEL_PATH, PREPROCESSOR_PATH)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    model_available = MODEL_PATH.is_file()
    preprocessor_available = PREPROCESSOR_PATH.is_file()
    return HealthResponse(
        status="ready" if model_available and preprocessor_available else "not_ready",
        model_artifact_available=model_available,
        preprocessor_artifact_available=preprocessor_available,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_price(features: HouseFeatures) -> PredictionResponse:
    if not MODEL_PATH.is_file() or not PREPROCESSOR_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are missing. Run the training pipeline first.",
        )

    feature_frame = pd.DataFrame([features.model_dump()])
    prediction = get_prediction_pipeline().predict(feature_frame)
    return PredictionResponse(predicted_price=float(prediction[0]))
