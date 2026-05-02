from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(slots=True)
class AppConfig:
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    model_dir: Path
    artifact_dir: Path
    random_state: int

    @classmethod
    def from_env(cls) -> AppConfig:
        model_dir = Path(os.getenv("MODEL_DIR", "models"))
        artifact_dir = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
        model_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
            mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "uplift-smoke"),
            model_dir=model_dir,
            artifact_dir=artifact_dir,
            random_state=int(os.getenv("RANDOM_STATE", "42")),
        )
