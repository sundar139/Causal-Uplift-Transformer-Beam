from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import yaml

from causal_uplift.baselines import SLearnerBaseline
from causal_uplift.config import AppConfig, DataConfig, TrainingConfig
from causal_uplift.data import (
    CriteoDataset,
    DatasetSplit,
    create_train_validation_test_split,
    load_criteo_dataset,
    resolve_materialization_paths,
)
from causal_uplift.evaluate import compute_uplift_metrics
from causal_uplift.mlflow_utils import initialize_mlflow, log_run_payload, start_named_run
from causal_uplift.preprocessing import NumericFeaturePreprocessor
from causal_uplift.search_space import suggest_s_learner_params, suggest_transformer_params
from causal_uplift.transformer import FTTransformerUpliftModel, TorchUpliftTrainer

PROJECT_NAME = "causal-uplift-transformer-beam"
MODEL_FAMILIES = ("ft_transformer", "s_learner_logistic")


@dataclass(slots=True)
class StudyConfig:
    study_name: str
    storage: str
    direction: str
    n_trials_transformer: int
    n_trials_s_learner: int
    timeout_seconds: int | None
    primary_metric: str
    tie_breaker_metric: str


@dataclass(slots=True)
class TuningRuntimeConfig:
    max_epochs: int
    early_stopping_patience: int
    num_workers: int


@dataclass(slots=True)
class TuningArtifactConfig:
    tuning_dir: Path
    best_params_name: str
    trials_name: str
    summary_name: str


@dataclass(slots=True)
class TuningConfig:
    random_state: int
    dataset_variant: str
    data: DataConfig
    study: StudyConfig
    training: TuningRuntimeConfig
    artifacts: TuningArtifactConfig


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def load_tuning_config(path: str | Path) -> TuningConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    data_payload = payload.get("data", {})
    study_payload = payload.get("study", {})
    training_payload = payload.get("training", {})
    artifacts_payload = payload.get("artifacts", {})

    data = DataConfig(
        sample_size=int(data_payload.get("sample_size", 0)),
        percent10=bool(data_payload.get("percent10", True)),
        test_size=float(data_payload.get("test_size", 0.2)),
        validation_size=float(data_payload.get("validation_size", 0.2)),
        target_col=str(data_payload.get("target_col", "conversion")),
        treatment_col=str(data_payload.get("treatment_col", "treatment")),
    )
    variant = str(payload.get("dataset_variant", data.dataset_variant))
    if variant != data.dataset_variant:
        raise ValueError(
            f"dataset_variant={variant!r} does not match percent10={data.percent10} "
            f"({data.dataset_variant!r})."
        )

    return TuningConfig(
        random_state=int(payload.get("random_state", 42)),
        dataset_variant=variant,
        data=data,
        study=StudyConfig(
            study_name=str(study_payload.get("study_name", f"uplift-{variant}-tuning")),
            storage=str(study_payload.get("storage", "sqlite:///optuna_studies.db")),
            direction=str(study_payload.get("direction", "maximize")),
            n_trials_transformer=int(study_payload.get("n_trials_transformer", 20)),
            n_trials_s_learner=int(study_payload.get("n_trials_s_learner", 20)),
            timeout_seconds=study_payload.get("timeout_seconds"),
            primary_metric=str(study_payload.get("primary_metric", "qini_auc")),
            tie_breaker_metric=str(study_payload.get("tie_breaker_metric", "policy_gain_top20")),
        ),
        training=TuningRuntimeConfig(
            max_epochs=int(training_payload.get("max_epochs", 12)),
            early_stopping_patience=int(training_payload.get("early_stopping_patience", 3)),
            num_workers=int(training_payload.get("num_workers", 0)),
        ),
        artifacts=TuningArtifactConfig(
            tuning_dir=Path(artifacts_payload.get("tuning_dir", f"artifacts/tuning/{variant}")),
            best_params_name=str(artifacts_payload.get("best_params_name", "best_params.json")),
            trials_name=str(artifacts_payload.get("trials_name", "optuna_trials.csv")),
            summary_name=str(artifacts_payload.get("summary_name", "tuning_summary.json")),
        ),
    )


def resolve_tuning_output_paths(config: TuningConfig) -> dict[str, Path]:
    tuning_dir = config.artifacts.tuning_dir
    return {
        "tuning_dir": tuning_dir,
        "best_params": tuning_dir / config.artifacts.best_params_name,
        "trials": tuning_dir / config.artifacts.trials_name,
        "summary": tuning_dir / config.artifacts.summary_name,
        "checkpoints": tuning_dir / "checkpoints",
    }


def _training_config_from_tuning(config: TuningConfig) -> TrainingConfig:
    return TrainingConfig(random_state=config.random_state, data=config.data)


def _split_from_processed_frame(
    frame: pd.DataFrame, config: TuningConfig
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    ignored = {config.data.target_col, config.data.treatment_col, "split"}
    feature_columns = [column for column in frame.columns if column not in ignored]
    return (
        frame[feature_columns].copy(),
        frame[config.data.target_col].astype(int).reset_index(drop=True),
        frame[config.data.treatment_col].astype(int).reset_index(drop=True),
    )


def _load_materialized_split_if_available(config: TuningConfig) -> DatasetSplit | None:
    paths = resolve_materialization_paths(_training_config_from_tuning(config))["processed_paths"]
    if not all(path.exists() for path in paths.values()):
        return None

    frames = {split_name: pd.read_parquet(path) for split_name, path in paths.items()}
    X_train, y_train, t_train = _split_from_processed_frame(frames["train"], config)
    X_validation, y_validation, t_validation = _split_from_processed_frame(
        frames["validation"],
        config,
    )
    X_test, y_test, t_test = _split_from_processed_frame(frames["test"], config)
    return DatasetSplit(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
        treatment_train=t_train,
        treatment_validation=t_validation,
        treatment_test=t_test,
    )


def load_tuning_split(config: TuningConfig) -> DatasetSplit:
    materialized = _load_materialized_split_if_available(config)
    if materialized is not None:
        return materialized

    dataset: CriteoDataset = load_criteo_dataset(
        sample_size=config.data.sample_size,
        percent10=config.data.percent10,
        target_col=config.data.target_col,
        treatment_col=config.data.treatment_col,
    )
    return create_train_validation_test_split(
        dataset,
        validation_size=config.data.validation_size,
        test_size=config.data.test_size,
        random_state=config.random_state,
    )


@dataclass(slots=True)
class PreparedTuningData:
    X_train_np: np.ndarray
    X_validation_np: np.ndarray
    X_train_df: pd.DataFrame
    X_validation_df: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    t_train: pd.Series
    t_validation: pd.Series


def prepare_tuning_data(split: DatasetSplit) -> PreparedTuningData:
    preprocessor = NumericFeaturePreprocessor()
    X_train_np = preprocessor.fit_transform(split.X_train)
    X_validation_np = preprocessor.transform(split.X_validation)
    feature_columns = preprocessor.feature_columns

    return PreparedTuningData(
        X_train_np=X_train_np,
        X_validation_np=X_validation_np,
        X_train_df=pd.DataFrame(X_train_np, columns=feature_columns),
        X_validation_df=pd.DataFrame(X_validation_np, columns=feature_columns),
        y_train=split.y_train.reset_index(drop=True),
        y_validation=split.y_validation.reset_index(drop=True),
        t_train=split.treatment_train.reset_index(drop=True),
        t_validation=split.treatment_validation.reset_index(drop=True),
    )


def _family_slug(model_family: str) -> str:
    return model_family.replace("_", "-")


def _metrics_for_prediction(
    prediction_uplift: np.ndarray,
    treatment_proba: np.ndarray,
    data: PreparedTuningData,
) -> dict[str, float]:
    return compute_uplift_metrics(
        y_true=data.y_validation.to_numpy(),
        treatment=data.t_validation.to_numpy(),
        uplift=prediction_uplift,
        treatment_proba=treatment_proba,
    )


def _objective_transformer(
    trial: optuna.trial.Trial,
    config: TuningConfig,
    data: PreparedTuningData,
    checkpoint_dir: Path,
) -> float:
    params = suggest_transformer_params(trial)
    checkpoint_path = checkpoint_dir / f"ft_transformer_trial_{trial.number:03d}.pt"

    model = FTTransformerUpliftModel(
        num_features=data.X_train_np.shape[1] + 1,
        d_token=int(params["embedding_dim"]),
        num_layers=int(params["num_layers"]),
        num_heads=int(params["num_heads"]),
        dropout=float(params["dropout"]),
        hidden_dim=int(params["hidden_dim"]),
    )
    trainer = TorchUpliftTrainer(
        model=model,
        learning_rate=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
        batch_size=int(params["batch_size"]),
        epochs=config.training.max_epochs,
        patience=config.training.early_stopping_patience,
        num_workers=config.training.num_workers,
        random_state=config.random_state,
    )

    train_inputs = np.concatenate(
        [data.X_train_np, data.t_train.to_numpy(dtype=np.float32).reshape(-1, 1)],
        axis=1,
    )
    validation_inputs = np.concatenate(
        [data.X_validation_np, data.t_validation.to_numpy(dtype=np.float32).reshape(-1, 1)],
        axis=1,
    )
    trainer.fit(
        X_train=train_inputs,
        y_train=data.y_train.to_numpy(dtype=np.float32),
        X_validation=validation_inputs,
        y_validation=data.y_validation.to_numpy(dtype=np.float32),
        checkpoint_path=checkpoint_path,
    )
    prediction = trainer.predict_uplift(data.X_validation_np)
    metrics = _metrics_for_prediction(prediction.uplift, prediction.treatment_proba, data)
    _log_trial_run("ft_transformer", trial.number, params, metrics, config)
    trial.set_user_attr("resolved_params", params)
    for metric_name, metric_value in metrics.items():
        trial.set_user_attr(metric_name, metric_value)
    return metrics[config.study.primary_metric]


def _objective_s_learner(
    trial: optuna.trial.Trial,
    config: TuningConfig,
    data: PreparedTuningData,
) -> float:
    params = suggest_s_learner_params(trial)
    model = SLearnerBaseline(random_state=config.random_state, **params)
    model.fit(data.X_train_df, data.y_train, data.t_train)
    prediction = model.predict_uplift(data.X_validation_df)
    metrics = _metrics_for_prediction(prediction.uplift, prediction.treatment_proba, data)
    _log_trial_run("s_learner_logistic", trial.number, params, metrics, config)
    trial.set_user_attr("resolved_params", params)
    for metric_name, metric_value in metrics.items():
        trial.set_user_attr(metric_name, metric_value)
    return metrics[config.study.primary_metric]


def _log_trial_run(
    model_family: str,
    trial_number: int,
    params: dict[str, Any],
    metrics: dict[str, float],
    config: TuningConfig,
) -> None:
    run_name = f"tuning-{_family_slug(model_family)}-trial-{trial_number:03d}"
    with start_named_run(
        run_name=run_name,
        tags={
            "project": PROJECT_NAME,
            "run_type": "tuning",
            "dataset_variant": config.dataset_variant,
            "model_family": model_family,
            "trial_number": str(trial_number),
            "primary_metric": config.study.primary_metric,
        },
    ):
        log_run_payload(
            params={
                **params,
                "dataset_variant": config.dataset_variant,
                "model_family": model_family,
                "max_epochs": config.training.max_epochs,
                "early_stopping_patience": config.training.early_stopping_patience,
            },
            metrics={
                "qini_auc": metrics["qini_auc"],
                "uplift_auc": metrics["uplift_auc"],
                "policy_gain_top20": metrics["policy_gain_top20"],
                "treatment_response_auc": metrics["treatment_response_auc"],
            },
            artifact_paths=[],
        )


def _create_study(config: TuningConfig, model_family: str) -> optuna.Study:
    return optuna.create_study(
        study_name=f"{config.study.study_name}-{_family_slug(model_family)}",
        storage=config.study.storage,
        direction=config.study.direction,
        load_if_exists=True,
    )


def _best_params_for_trial(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    resolved = trial.user_attrs.get("resolved_params")
    if isinstance(resolved, dict):
        return resolved
    return dict(trial.params)


def _best_trial_payload(
    study: optuna.Study,
    model_family: str,
    primary_metric: str,
    tie_breaker_metric: str,
) -> dict[str, Any]:
    best_trial = study.best_trial
    return {
        "model_family": model_family,
        "best_value": float(best_trial.value or 0.0),
        "tie_breaker_value": float(best_trial.user_attrs.get(tie_breaker_metric, 0.0)),
        "best_params": _best_params_for_trial(best_trial),
        "primary_metric": primary_metric,
        "tie_breaker_metric": tie_breaker_metric,
    }


def build_best_params_payload(
    config: TuningConfig,
    model_payloads: dict[str, dict[str, Any]],
    generated_at_utc: str,
) -> dict[str, Any]:
    ordered = sorted(
        model_payloads.values(),
        key=lambda payload: (
            float(payload["best_value"]),
            float(payload["tie_breaker_value"]),
        ),
        reverse=True,
    )
    return {
        "dataset_variant": config.dataset_variant,
        "selection_metric": config.study.primary_metric,
        "tie_breaker_metric": config.study.tie_breaker_metric,
        "best_model_family": ordered[0]["model_family"],
        "models": {
            family: {
                "best_value": payload["best_value"],
                "best_params": payload["best_params"],
            }
            for family, payload in model_payloads.items()
        },
        "generated_at_utc": generated_at_utc,
    }


def build_tuning_summary_payload(
    config: TuningConfig,
    studies: dict[str, optuna.Study],
    best_params_payload: dict[str, Any],
    paths: dict[str, Path],
    generated_at_utc: str,
) -> dict[str, Any]:
    completed_counts = {
        family: len(
            [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        )
        for family, study in studies.items()
    }
    return {
        "dataset_variant": config.dataset_variant,
        "study_names": {family: study.study_name for family, study in studies.items()},
        "n_trials_requested": {
            "ft_transformer": config.study.n_trials_transformer,
            "s_learner_logistic": config.study.n_trials_s_learner,
        },
        "n_trials_completed": completed_counts,
        "best_model_family": best_params_payload["best_model_family"],
        "best_qini_auc": best_params_payload["models"][best_params_payload["best_model_family"]][
            "best_value"
        ],
        "best_params_path": str(paths["best_params"]),
        "trials_path": str(paths["trials"]),
        "generated_at_utc": generated_at_utc,
    }


def _trials_frame(studies: dict[str, optuna.Study]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for family, study in studies.items():
        frame = study.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs"))
        frame.insert(0, "model_family", family)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_tuning_artifacts(
    config: TuningConfig,
    studies: dict[str, optuna.Study],
    paths: dict[str, Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    generated_at_utc = _utc_now()
    paths["tuning_dir"].mkdir(parents=True, exist_ok=True)
    trial_payloads = {
        family: _best_trial_payload(
            study,
            family,
            config.study.primary_metric,
            config.study.tie_breaker_metric,
        )
        for family, study in studies.items()
    }
    best_params_payload = build_best_params_payload(config, trial_payloads, generated_at_utc)
    summary_payload = build_tuning_summary_payload(
        config,
        studies,
        best_params_payload,
        paths,
        generated_at_utc,
    )

    _trials_frame(studies).to_csv(paths["trials"], index=False)
    with paths["best_params"].open("w", encoding="utf-8") as handle:
        json.dump(best_params_payload, handle, indent=2)
    with paths["summary"].open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    return best_params_payload, summary_payload


def run_tuning(config_path: str | Path) -> dict[str, Any]:
    app_config = AppConfig.from_env()
    config = load_tuning_config(config_path)
    initialize_mlflow(app_config)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    split = load_tuning_split(config)
    data = prepare_tuning_data(split)
    paths = resolve_tuning_output_paths(config)
    paths["checkpoints"].mkdir(parents=True, exist_ok=True)

    studies = {
        "ft_transformer": _create_study(config, "ft_transformer"),
        "s_learner_logistic": _create_study(config, "s_learner_logistic"),
    }
    studies["ft_transformer"].optimize(
        lambda trial: _objective_transformer(trial, config, data, paths["checkpoints"]),
        n_trials=config.study.n_trials_transformer,
        timeout=config.study.timeout_seconds,
    )
    studies["s_learner_logistic"].optimize(
        lambda trial: _objective_s_learner(trial, config, data),
        n_trials=config.study.n_trials_s_learner,
        timeout=config.study.timeout_seconds,
    )

    best_params_payload, summary_payload = write_tuning_artifacts(config, studies, paths)
    with start_named_run(
        run_name=f"tuning-summary-{config.dataset_variant}",
        tags={
            "project": PROJECT_NAME,
            "run_type": "tuning",
            "dataset_variant": config.dataset_variant,
            "primary_metric": config.study.primary_metric,
        },
    ):
        log_run_payload(
            params={
                "dataset_variant": config.dataset_variant,
                "best_model_family": best_params_payload["best_model_family"],
                "n_trials_transformer": config.study.n_trials_transformer,
                "n_trials_s_learner": config.study.n_trials_s_learner,
            },
            metrics={"best_qini_auc": float(summary_payload["best_qini_auc"])},
            artifact_paths=[paths["best_params"], paths["trials"], paths["summary"]],
        )

    print(json.dumps(summary_payload, indent=2))
    return summary_payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optuna tuning for uplift models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tuning.yaml",
        help="Path to tuning configuration",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_tuning(config_path=args.config)


if __name__ == "__main__":
    main()
