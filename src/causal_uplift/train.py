from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow

from causal_uplift.baselines import TwoModelUpliftBaseline
from causal_uplift.config import AppConfig
from causal_uplift.data import load_criteo_sample
from causal_uplift.evaluate import compute_uplift_metrics


def run_smoke_training(sample_size: int = 10_000) -> dict[str, str | dict[str, float]]:
    config = AppConfig.from_env()
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    dataset = load_criteo_sample(sample_size=sample_size, random_state=config.random_state)

    model = TwoModelUpliftBaseline(random_state=config.random_state)
    model.fit(dataset.X_train, dataset.y_train, dataset.treatment_train)

    predictions = model.predict_uplift(dataset.X_test)
    metrics = compute_uplift_metrics(
        y_true=dataset.y_test.to_numpy(),
        treatment=dataset.treatment_test.to_numpy(),
        uplift=predictions.uplift,
        treatment_proba=predictions.treatment_proba,
    )

    model_path = Path(config.model_dir) / "smoke_uplift_baseline.joblib"
    metrics_path = Path(config.artifact_dir) / "smoke_metrics.json"

    with mlflow.start_run(run_name="smoke-baseline"):
        mlflow.log_params(
            {
                "sample_size": sample_size,
                "random_state": config.random_state,
                "model": "two_model_logistic_regression",
            }
        )
        mlflow.set_tags(
            {
                "pipeline": "smoke",
                "component": "baseline",
            }
        )
        mlflow.log_metrics(metrics)

        joblib.dump(model, model_path)
        with metrics_path.open("w", encoding="utf-8") as file_handle:
            json.dump(metrics, file_handle, indent=2)

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(metrics_path))

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline uplift smoke training.")
    parser.add_argument(
        "--sample-size", type=int, default=10_000, help="Number of rows for smoke run"
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    result = run_smoke_training(sample_size=args.sample_size)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
