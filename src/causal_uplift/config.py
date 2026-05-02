from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
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

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        if tracking_uri.startswith("sqlite:///"):
            sqlite_path = Path(tracking_uri.replace("sqlite:///", "", 1))
            if sqlite_path.parent != Path("."):
                sqlite_path.parent.mkdir(parents=True, exist_ok=True)

        (artifact_dir / "evaluation").mkdir(parents=True, exist_ok=True)

        return cls(
            mlflow_tracking_uri=tracking_uri,
            mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "causal-uplift-training"),
            model_dir=model_dir,
            artifact_dir=artifact_dir,
            random_state=int(os.getenv("RANDOM_STATE", "42")),
        )


@dataclass(slots=True)
class SplitConfig:
    validation_size: float = 0.15
    test_size: float = 0.15


@dataclass(slots=True)
class TransformerConfig:
    d_token: int = 32
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    epochs: int = 6
    patience: int = 2


@dataclass(slots=True)
class TrainingConfig:
    sample_size: int = 0
    split: SplitConfig = field(default_factory=SplitConfig)
    models: list[str] = field(
        default_factory=lambda: [
            "two_model_logistic",
            "s_learner_logistic",
            "t_learner_logistic",
            "ft_transformer",
        ]
    )
    transformer: TransformerConfig = field(default_factory=TransformerConfig)


def load_training_config(path: str | Path) -> TrainingConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    split_payload = payload.get("split", {})
    transformer_payload = payload.get("transformer", {})

    split = SplitConfig(
        validation_size=float(split_payload.get("validation_size", 0.15)),
        test_size=float(split_payload.get("test_size", 0.15)),
    )
    if (
        split.validation_size <= 0
        or split.test_size <= 0
        or (split.validation_size + split.test_size) >= 1
    ):
        raise ValueError("validation_size and test_size must be positive and their sum must be < 1")

    transformer = TransformerConfig(
        d_token=int(transformer_payload.get("d_token", 32)),
        num_layers=int(transformer_payload.get("num_layers", 2)),
        num_heads=int(transformer_payload.get("num_heads", 4)),
        dropout=float(transformer_payload.get("dropout", 0.1)),
        learning_rate=float(transformer_payload.get("learning_rate", 1e-3)),
        weight_decay=float(transformer_payload.get("weight_decay", 1e-4)),
        batch_size=int(transformer_payload.get("batch_size", 1024)),
        epochs=int(transformer_payload.get("epochs", 6)),
        patience=int(transformer_payload.get("patience", 2)),
    )

    models = payload.get(
        "models",
        [
            "two_model_logistic",
            "s_learner_logistic",
            "t_learner_logistic",
            "ft_transformer",
        ],
    )

    return TrainingConfig(
        sample_size=int(payload.get("sample_size", 0)),
        split=split,
        models=[str(name) for name in models],
        transformer=transformer,
    )
