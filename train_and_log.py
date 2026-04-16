from __future__ import annotations

import os
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from model_utils import MODEL_INFO_PATH, save_model_bundle, train_and_select_best_model

EXPERIMENT_NAME = "taxi-tip-prediction"
REGISTERED_MODEL_NAME = "taxi-tip-regressor"


def configure_mlflow() -> None:
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"file:{(Path(__file__).resolve().parent / 'mlruns').as_posix()}",
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def main() -> None:
    configure_mlflow()
    training_result = train_and_select_best_model()
    client = MlflowClient()

    best_run_id = None
    best_model_uri = None
    best_rmse = None

    for result in training_result["all_results"]:
        model_name = result["model_name"]
        pipeline = result["pipeline"]
        metrics = result["metrics"]

        with mlflow.start_run(run_name=model_name) as run:
            params = pipeline.named_steps["model"].get_params()
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tags(
                {
                    "model_type": type(pipeline.named_steps["model"]).__name__,
                    "dataset_version": training_result["dataset_version"],
                }
            )

            signature = infer_signature(
                training_result["sample_input"],
                pipeline.predict(training_result["sample_input"]),
            )

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                signature=signature,
            )

            if best_rmse is None or metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_run_id = run.info.run_id
                best_model_uri = f"runs:/{run.info.run_id}/model"

    if not best_run_id or not best_model_uri:
        raise RuntimeError("No MLflow runs were created.")

    model_version = mlflow.register_model(best_model_uri, REGISTERED_MODEL_NAME)
    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=model_version.version,
        description=(
            "Best Assignment 4 regression model. "
            f"RMSE={training_result['best_metrics']['rmse']:.4f}, "
            f"MAE={training_result['best_metrics']['mae']:.4f}, "
            f"R2={training_result['best_metrics']['r2']:.4f}"
        ),
    )

    save_model_bundle(
        pipeline=training_result["best_pipeline"],
        metrics=training_result["best_metrics"],
        model_name=REGISTERED_MODEL_NAME,
        model_version=str(model_version.version),
        extra_metadata={
            "mlflow_tracking_uri": mlflow.get_tracking_uri(),
            "registered_model_name": REGISTERED_MODEL_NAME,
            "model_info_path": str(MODEL_INFO_PATH),
        },
    )

    print("Best model:", training_result["best_model_name"])
    print("Metrics:", training_result["best_metrics"])
    print("Registered model version:", model_version.version)
    print("Tracking URI:", mlflow.get_tracking_uri())


if __name__ == "__main__":
    main()
