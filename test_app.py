from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURE_COLUMNS = [
    "pickup_hour",
    "pickup_day_of_week",
    "is_weekend",
    "trip_duration_minutes",
    "trip_speed_mph",
    "log_trip_distance",
    "fare_per_mile",
    "fare_per_minute",
    "pickup_borough",
    "dropoff_borough",
    "passenger_count",
    "trip_distance",
    "fare_amount",
]


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    model_path = tmp_path / "taxi_tip_model.joblib"
    metadata_path = tmp_path / "model_info.json"

    training_frame = pd.DataFrame(
        [
            {
                "pickup_hour": 8,
                "pickup_day_of_week": 1,
                "is_weekend": False,
                "trip_duration_minutes": 12.5,
                "trip_speed_mph": 14.4,
                "log_trip_distance": 1.1,
                "fare_per_mile": 3.2,
                "fare_per_minute": 0.72,
                "pickup_borough": "Manhattan",
                "dropoff_borough": "Queens",
                "passenger_count": 1,
                "trip_distance": 3.0,
                "fare_amount": 9.0,
            },
            {
                "pickup_hour": 19,
                "pickup_day_of_week": 5,
                "is_weekend": True,
                "trip_duration_minutes": 28.0,
                "trip_speed_mph": 16.2,
                "log_trip_distance": 1.8,
                "fare_per_mile": 4.0,
                "fare_per_minute": 0.95,
                "pickup_borough": "Brooklyn",
                "dropoff_borough": "Manhattan",
                "passenger_count": 2,
                "trip_distance": 7.5,
                "fare_amount": 30.0,
            },
            {
                "pickup_hour": 13,
                "pickup_day_of_week": 2,
                "is_weekend": False,
                "trip_duration_minutes": 22.0,
                "trip_speed_mph": 12.8,
                "log_trip_distance": 1.5,
                "fare_per_mile": 3.6,
                "fare_per_minute": 0.81,
                "pickup_borough": "Queens",
                "dropoff_borough": "Queens",
                "passenger_count": 3,
                "trip_distance": 4.7,
                "fare_amount": 17.8,
            },
        ]
    )
    targets = pd.Series([1.9, 6.4, 3.5])

    numeric_features = [
        "pickup_hour",
        "pickup_day_of_week",
        "is_weekend",
        "trip_duration_minutes",
        "trip_speed_mph",
        "log_trip_distance",
        "fare_per_mile",
        "fare_per_minute",
        "passenger_count",
        "trip_distance",
        "fare_amount",
    ]
    categorical_features = ["pickup_borough", "dropoff_borough"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=20, random_state=42)),
        ]
    )
    model.fit(training_frame, targets)
    joblib.dump(model, model_path)

    metadata_path.write_text(
        json.dumps(
            {
                "model_name": "taxi-tip-regressor",
                "model_version": "test-1",
                "feature_names": FEATURE_COLUMNS,
                "training_metrics": {"mae": 0.1, "rmse": 0.2, "r2": 0.95},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("MODEL_INFO_PATH", str(metadata_path))

    sys.modules.pop("app", None)
    app_module = importlib.import_module("app")

    with TestClient(app_module.app) as test_client:
        yield test_client


def valid_payload() -> dict[str, object]:
    return {
        "pickup_hour": 10,
        "pickup_day_of_week": 3,
        "is_weekend": False,
        "trip_duration_minutes": 15.0,
        "trip_speed_mph": 13.0,
        "log_trip_distance": 1.2,
        "fare_per_mile": 3.5,
        "fare_per_minute": 0.8,
        "pickup_borough": "Manhattan",
        "dropoff_borough": "Brooklyn",
        "passenger_count": 2,
        "trip_distance": 3.8,
        "fare_amount": 13.4,
    }


def test_single_prediction_success(client: TestClient) -> None:
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "test-1"
    assert "prediction_id" in body
    assert isinstance(body["predicted_tip_amount"], float)


def test_batch_prediction_success(client: TestClient) -> None:
    response = client.post("/predict/batch", json=[valid_payload(), valid_payload()])
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "test-1"
    assert len(body["predictions"]) == 2


def test_invalid_input_rejected(client: TestClient) -> None:
    payload = valid_payload()
    payload["pickup_hour"] = 24
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_health_check(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": "test-1",
    }


def test_model_info(client: TestClient) -> None:
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "taxi-tip-regressor"
    assert body["model_version"] == "test-1"
    assert len(body["feature_names"]) == len(FEATURE_COLUMNS)


def test_zero_distance_trip_rejected(client: TestClient) -> None:
    payload = valid_payload()
    payload["trip_distance"] = 0
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_batch_size_limit_enforced(client: TestClient) -> None:
    response = client.post("/predict/batch", json=[valid_payload()] * 101)
    assert response.status_code == 422
