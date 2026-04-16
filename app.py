from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from model_utils import resolve_model_paths


class TripFeatures(BaseModel):
    pickup_hour: int = Field(..., ge=0, le=23)
    pickup_day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: bool
    trip_duration_minutes: float = Field(..., gt=0, le=300)
    trip_speed_mph: float = Field(..., gt=0, le=100)
    log_trip_distance: float = Field(..., ge=0, le=10)
    fare_per_mile: float = Field(..., ge=0, le=200)
    fare_per_minute: float = Field(..., ge=0, le=50)
    pickup_borough: str = Field(..., min_length=1, max_length=64)
    dropoff_borough: str = Field(..., min_length=1, max_length=64)
    passenger_count: int = Field(..., ge=1, le=8)
    trip_distance: float = Field(..., gt=0, le=200)
    fare_amount: float = Field(..., ge=0, le=500)


class PredictionResponse(BaseModel):
    prediction_id: str
    model_version: str
    predicted_tip_amount: float


class BatchPredictionResponse(BaseModel):
    model_version: str
    predictions: list[PredictionResponse]


class ErrorResponse(BaseModel):
    error: str
    message: str


def load_model_artifacts() -> dict[str, Any]:
    model_path, model_info_path = resolve_model_paths()

    if not model_path.exists():
        raise RuntimeError(f"Model file not found: {model_path}")
    if not model_info_path.exists():
        raise RuntimeError(f"Model metadata file not found: {model_info_path}")

    model = joblib.load(model_path)
    metadata = json.loads(model_info_path.read_text(encoding="utf-8"))
    return {
        "model": model,
        "metadata": metadata,
        "model_path": str(Path(model_path).resolve()),
        "model_info_path": str(Path(model_info_path).resolve()),
    }


def make_prediction_record(payload: TripFeatures, loaded_state: dict[str, Any]) -> PredictionResponse:
    features = pd.DataFrame([payload.model_dump()])
    prediction = float(loaded_state["model"].predict(features)[0])
    return PredictionResponse(
        prediction_id=str(uuid.uuid4()),
        model_version=loaded_state["metadata"]["model_version"],
        predicted_tip_amount=round(prediction, 2),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.loaded_state = load_model_artifacts()
    yield

# Create a FastAPI application in a file named app.py with the following structure: Model Loading, Single Prediction Endpoint, Response Format, Response Format, Input Validation
# Extend your API with the following endpoints: Batch Prediction, Health Check, Model Info and Error Handling     
def create_app() -> FastAPI:
    app = FastAPI(
        title="Taxi Tip Prediction API",
        version="1.0.0",
        description="Assignment 4 FastAPI service for serving taxi tip predictions.",
        lifespan=lifespan,
    )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_server_error",
                message="The server could not complete the request.",
            ).model_dump(),
        )
    
    @app.get("/")
    async def root() -> dict[str, str]:
        return {"message": "API is running"}

    @app.get("/health")
    async def health() -> dict[str, Any]:
        loaded_state = getattr(app.state, "loaded_state", None)
        return {
            "status": "ok",
            "model_loaded": loaded_state is not None,
            "model_version": loaded_state["metadata"]["model_version"]
            if loaded_state is not None
            else None,
        }

    @app.get("/model/info")
    async def model_info() -> dict[str, Any]:
        loaded_state = app.state.loaded_state
        metadata = loaded_state["metadata"]
        return {
            "model_name": metadata["model_name"],
            "model_version": metadata["model_version"],
            "feature_names": metadata["feature_names"],
            "excluded_leakage_columns": metadata.get("excluded_leakage_columns", []),
            "training_metrics": metadata["training_metrics"],
            "model_path": loaded_state["model_path"],
        }

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(payload: TripFeatures) -> PredictionResponse:
        return make_prediction_record(payload, app.state.loaded_state)

    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    async def predict_batch(payload: list[TripFeatures]) -> BatchPredictionResponse:
        if not payload:
            raise HTTPException(status_code=422, detail="At least one record is required.")
        if len(payload) > 100:
            raise HTTPException(
                status_code=422,
                detail="Batch prediction supports at most 100 records.",
            )

        predictions = [
            make_prediction_record(record, app.state.loaded_state) for record in payload
        ]
        return BatchPredictionResponse(
            model_version=app.state.loaded_state["metadata"]["model_version"],
            predictions=predictions,
        )

    return app


app = create_app()
