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
class DataConfig:
    sample_size: int = 0
    percent10: bool = True
    test_size: float = 0.15
    validation_size: float = 0.15
    target_col: str = "conversion"
    treatment_col: str = "treatment"

    @property
    def dataset_variant(self) -> str:
        return "percent10" if self.percent10 else "full"


@dataclass(slots=True)
class TrainingRuntimeConfig:
    batch_size: int = 1024
    max_epochs: int = 6
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 2
    num_workers: int = 0


@dataclass(slots=True)
class TransformerConfig:
    embedding_dim: int = 32
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    hidden_dim: int = 128

    @property
    def d_token(self) -> int:
        return self.embedding_dim


@dataclass(slots=True)
class ArtifactConfig:
    best_model_name: str = "best_transformer_uplift.pt"
    metrics_name: str = "full_training_metrics.json"
    predictions_name: str = "test_predictions.csv"


@dataclass(slots=True)
class TrainingConfig:
    random_state: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingRuntimeConfig = field(default_factory=TrainingRuntimeConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    models: list[str] = field(
        default_factory=lambda: [
            "two_model_logistic",
            "s_learner_logistic",
            "t_learner_logistic",
            "ft_transformer",
        ]
    )

    @property
    def sample_size(self) -> int:
        return self.data.sample_size

    @property
    def split(self) -> SplitConfig:
        return SplitConfig(
            validation_size=self.data.validation_size,
            test_size=self.data.test_size,
        )


def _validate_split(validation_size: float, test_size: float) -> None:
    if validation_size <= 0 or test_size <= 0 or (validation_size + test_size) >= 1:
        raise ValueError("validation_size and test_size must be positive and their sum must be < 1")


def _load_new_schema(payload: dict[str, object]) -> TrainingConfig:
    data_payload = payload.get("data", {})
    training_payload = payload.get("training", {})
    transformer_payload = payload.get("transformer", {})
    artifacts_payload = payload.get("artifacts", {})

    data = DataConfig(
        sample_size=int(data_payload.get("sample_size", 0)),
        percent10=bool(data_payload.get("percent10", True)),
        test_size=float(data_payload.get("test_size", 0.15)),
        validation_size=float(data_payload.get("validation_size", 0.15)),
        target_col=str(data_payload.get("target_col", "conversion")),
        treatment_col=str(data_payload.get("treatment_col", "treatment")),
    )
    _validate_split(data.validation_size, data.test_size)

    training = TrainingRuntimeConfig(
        batch_size=int(training_payload.get("batch_size", 1024)),
        max_epochs=int(training_payload.get("max_epochs", 6)),
        learning_rate=float(training_payload.get("learning_rate", 1e-3)),
        weight_decay=float(training_payload.get("weight_decay", 1e-4)),
        early_stopping_patience=int(training_payload.get("early_stopping_patience", 2)),
        num_workers=int(training_payload.get("num_workers", 0)),
    )

    transformer = TransformerConfig(
        embedding_dim=int(transformer_payload.get("embedding_dim", 32)),
        num_layers=int(transformer_payload.get("num_layers", 2)),
        num_heads=int(transformer_payload.get("num_heads", 4)),
        dropout=float(transformer_payload.get("dropout", 0.1)),
        hidden_dim=int(transformer_payload.get("hidden_dim", 128)),
    )

    artifacts = ArtifactConfig(
        best_model_name=str(artifacts_payload.get("best_model_name", "best_transformer_uplift.pt")),
        metrics_name=str(artifacts_payload.get("metrics_name", "full_training_metrics.json")),
        predictions_name=str(artifacts_payload.get("predictions_name", "test_predictions.csv")),
    )

    models = payload.get(
        "models",
        ["two_model_logistic", "s_learner_logistic", "t_learner_logistic", "ft_transformer"],
    )

    return TrainingConfig(
        random_state=int(payload.get("random_state", 42)),
        data=data,
        training=training,
        transformer=transformer,
        artifacts=artifacts,
        models=[str(name) for name in models],
    )


def _load_legacy_schema(payload: dict[str, object]) -> TrainingConfig:
    split_payload = payload.get("split", {})
    transformer_payload = payload.get("transformer", {})

    data = DataConfig(
        sample_size=int(payload.get("sample_size", 0)),
        percent10=True,
        test_size=float(split_payload.get("test_size", 0.15)),
        validation_size=float(split_payload.get("validation_size", 0.15)),
        target_col="conversion",
        treatment_col="treatment",
    )
    _validate_split(data.validation_size, data.test_size)

    training = TrainingRuntimeConfig(
        batch_size=int(transformer_payload.get("batch_size", 1024)),
        max_epochs=int(transformer_payload.get("epochs", 6)),
        learning_rate=float(transformer_payload.get("learning_rate", 1e-3)),
        weight_decay=float(transformer_payload.get("weight_decay", 1e-4)),
        early_stopping_patience=int(transformer_payload.get("patience", 2)),
        num_workers=0,
    )

    transformer = TransformerConfig(
        embedding_dim=int(transformer_payload.get("d_token", 32)),
        num_layers=int(transformer_payload.get("num_layers", 2)),
        num_heads=int(transformer_payload.get("num_heads", 4)),
        dropout=float(transformer_payload.get("dropout", 0.1)),
        hidden_dim=int(transformer_payload.get("d_token", 32)) * 4,
    )

    models = payload.get(
        "models",
        ["two_model_logistic", "s_learner_logistic", "t_learner_logistic", "ft_transformer"],
    )

    return TrainingConfig(
        random_state=int(payload.get("random_state", 42)),
        data=data,
        training=training,
        transformer=transformer,
        artifacts=ArtifactConfig(),
        models=[str(name) for name in models],
    )


def load_training_config(path: str | Path) -> TrainingConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if "data" in payload or "training" in payload or "artifacts" in payload:
        return _load_new_schema(payload)
    return _load_legacy_schema(payload)
