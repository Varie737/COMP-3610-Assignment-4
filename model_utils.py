from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

TRIP_DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
RAW_TRIP_PATH = RAW_DIR / "yellow_tripdata_2024-01.parquet"
ZONE_LOOKUP_PATH = RAW_DIR / "taxi_zone_lookup.csv"
MODEL_PATH = MODELS_DIR / "taxi_tip_model.joblib"
MODEL_INFO_PATH = MODELS_DIR / "model_info.json"

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

# These fields reveal or directly encode the target and must never be used as model inputs.
LEAKAGE_COLUMNS = {
    "tip_amount",
    "tip_amount_target",
    "high_tip",
    "total_amount",
}

NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = ["pickup_borough", "dropoff_borough"]


@dataclass
class DatasetBundle:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def ensure_directories() -> None:
    for path in (DATA_DIR, RAW_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path) -> Path:
    ensure_directories()
    if destination.exists():
        return destination

    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        handle.write(response.read())
    return destination


def engineer_assignment2_features(df: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working = working[working["payment_type"] == 1].copy()

    working["tpep_pickup_datetime"] = pd.to_datetime(
        working["tpep_pickup_datetime"], errors="coerce"
    )
    working["tpep_dropoff_datetime"] = pd.to_datetime(
        working["tpep_dropoff_datetime"], errors="coerce"
    )

    working = working.dropna(
        subset=[
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "PULocationID",
            "DOLocationID",
            "fare_amount",
            "trip_distance",
            "tip_amount",
            "passenger_count",
        ]
    )

    working = working[
        (working["trip_distance"] > 0)
        & (working["fare_amount"] >= 0)
        & (working["fare_amount"] <= 500)
        & (working["tpep_dropoff_datetime"] >= working["tpep_pickup_datetime"])
    ].copy()

    working["pickup_hour"] = working["tpep_pickup_datetime"].dt.hour
    working["pickup_day_of_week"] = working["tpep_pickup_datetime"].dt.dayofweek
    working["is_weekend"] = working["pickup_day_of_week"] >= 5

    duration_seconds = (
        working["tpep_dropoff_datetime"] - working["tpep_pickup_datetime"]
    ).dt.total_seconds()
    working["trip_duration_minutes"] = duration_seconds / 60.0
    working = working[working["trip_duration_minutes"] > 0].copy()

    duration_hours = working["trip_duration_minutes"] / 60.0
    working["trip_speed_mph"] = working["trip_distance"] / duration_hours
    working["log_trip_distance"] = np.log1p(working["trip_distance"])
    working["fare_per_mile"] = working["fare_amount"] / working["trip_distance"]
    working["fare_per_minute"] = (
        working["fare_amount"] / working["trip_duration_minutes"]
    )

    borough_lookup = zones[["LocationID", "Borough"]].copy()
    working = working.merge(
        borough_lookup.rename(
            columns={"LocationID": "PULocationID", "Borough": "pickup_borough"}
        ),
        on="PULocationID",
        how="left",
    )
    working = working.merge(
        borough_lookup.rename(
            columns={"LocationID": "DOLocationID", "Borough": "dropoff_borough"}
        ),
        on="DOLocationID",
        how="left",
    )

    working["pickup_borough"] = working["pickup_borough"].fillna("Unknown")
    working["dropoff_borough"] = working["dropoff_borough"].fillna("Unknown")

    return working


def load_assignment_data(sample_size: int | None = 200_000) -> pd.DataFrame:
    ensure_directories()
    download_file(TRIP_DATA_URL, RAW_TRIP_PATH)
    download_file(ZONE_LOOKUP_URL, ZONE_LOOKUP_PATH)

    df = pd.read_parquet(RAW_TRIP_PATH)
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_STATE)
    zones = pd.read_csv(ZONE_LOOKUP_PATH)
    return engineer_assignment2_features(df, zones)


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def create_train_test_split(df: pd.DataFrame) -> DatasetBundle:
    leakage_in_features = sorted(set(FEATURE_COLUMNS) & LEAKAGE_COLUMNS)
    if leakage_in_features:
        raise ValueError(
            f"Leakage-prone columns found in FEATURE_COLUMNS: {leakage_in_features}"
        )

    X = df[FEATURE_COLUMNS].copy()
    y = df["tip_amount"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    return DatasetBundle(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def build_regression_models() -> dict[str, Pipeline]:
    return {
        "linear_regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", LinearRegression()),
            ]
        ),
        "random_forest_regressor": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=24,
                        min_samples_split=10,
                        min_samples_leaf=3,
                        max_features=0.7,
                        bootstrap=True,
                        max_samples=0.5,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_regression_model(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    predictions = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_and_select_best_model(sample_size: int | None = 200_000) -> dict[str, Any]:
    df = load_assignment_data(sample_size=sample_size)
    dataset = create_train_test_split(df)

    model_results: list[dict[str, Any]] = []
    for model_name, pipeline in build_regression_models().items():
        pipeline.fit(dataset.X_train, dataset.y_train)
        metrics = evaluate_regression_model(pipeline, dataset.X_test, dataset.y_test)
        model_results.append(
            {"model_name": model_name, "pipeline": pipeline, "metrics": metrics}
        )

    best_result = min(model_results, key=lambda result: result["metrics"]["rmse"])
    return {
        "best_model_name": best_result["model_name"],
        "best_pipeline": best_result["pipeline"],
        "best_metrics": best_result["metrics"],
        "all_results": model_results,
        "feature_names": FEATURE_COLUMNS,
        "dataset_version": RAW_TRIP_PATH.name,
        "sample_input": dataset.X_test.head(5).copy(),
    }


def save_model_bundle(
    pipeline: Pipeline,
    metrics: dict[str, float],
    model_name: str,
    model_version: str = "local-dev",
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    ensure_directories()
    joblib.dump(pipeline, MODEL_PATH)

    payload: dict[str, Any] = {
        "model_name": model_name,
        "model_version": model_version,
        "feature_names": FEATURE_COLUMNS,
        "excluded_leakage_columns": sorted(LEAKAGE_COLUMNS),
        "training_metrics": metrics,
    }
    if extra_metadata:
        payload.update(extra_metadata)

    MODEL_INFO_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_saved_metadata(path: Path | None = None) -> dict[str, Any]:
    metadata_path = path or MODEL_INFO_PATH
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def resolve_model_paths() -> tuple[Path, Path]:
    model_path = Path(os.getenv("MODEL_PATH", str(MODEL_PATH)))
    model_info_path = Path(os.getenv("MODEL_INFO_PATH", str(MODEL_INFO_PATH)))
    return model_path, model_info_path
