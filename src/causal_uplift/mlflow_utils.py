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


@contextmanager
def start_named_run(run_name: str, tags: dict[str, str] | None = None) -> Iterator[None]:
    with mlflow.start_run(run_name=run_name):
        if tags:
            mlflow.set_tags(tags)
        yield


def log_artifacts(artifact_paths: list[Path], artifact_path: str | None = None) -> None:
    for artifact in artifact_paths:
        mlflow.log_artifact(str(artifact), artifact_path=artifact_path)


def log_run_payload(
    params: dict[str, int | float | str],
    metrics: dict[str, float],
    artifact_paths: list[Path],
) -> None:
    if params:
        mlflow.log_params(params)
    if metrics:
        mlflow.log_metrics(metrics)
    if artifact_paths:
        log_artifacts(artifact_paths)
