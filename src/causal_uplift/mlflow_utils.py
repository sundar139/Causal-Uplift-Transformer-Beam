from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import mlflow

from causal_uplift.config import AppConfig


def initialize_mlflow(config: AppConfig) -> None:
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)


@contextmanager
def start_model_run(model_name: str, tags: dict[str, str] | None = None) -> Iterator[None]:
    with mlflow.start_run(run_name=model_name):
        merged_tags = {"model_name": model_name}
        if tags:
            merged_tags.update(tags)
        mlflow.set_tags(merged_tags)
        yield


def log_run_payload(
    params: dict[str, int | float | str],
    metrics: dict[str, float],
    artifact_paths: list[Path],
) -> None:
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    for artifact in artifact_paths:
        mlflow.log_artifact(str(artifact))
