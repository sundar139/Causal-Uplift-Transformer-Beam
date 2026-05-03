from __future__ import annotations

import json
from pathlib import Path

import optuna

from causal_uplift.config import DataConfig, TrainingConfig
from causal_uplift.train import resolve_best_params_path
from causal_uplift.tuning import (
    build_best_params_payload,
    build_tuning_summary_payload,
    load_tuning_config,
    resolve_tuning_output_paths,
    write_tuning_artifacts,
)


def _completed_study(name: str, value: float, policy_gain: float) -> optuna.Study:
    study = optuna.create_study(study_name=name, direction="maximize")
    trial = study.ask()
    trial.set_user_attr("resolved_params", {"C": 1.0, "max_iter": 500})
    trial.set_user_attr("policy_gain_top20", policy_gain)
    study.tell(trial, value)
    return study


def test_best_params_payload_schema() -> None:
    config = load_tuning_config("configs/tuning.yaml")
    payload = build_best_params_payload(
        config,
        {
            "ft_transformer": {
                "model_family": "ft_transformer",
                "best_value": 0.11,
                "tie_breaker_value": 0.02,
                "best_params": {"embedding_dim": 64, "num_heads": 4},
            },
            "s_learner_logistic": {
                "model_family": "s_learner_logistic",
                "best_value": 0.12,
                "tie_breaker_value": 0.01,
                "best_params": {"C": 1.0, "penalty": "l2", "class_weight": None, "max_iter": 500},
            },
        },
        generated_at_utc="2026-05-03T00:00:00+00:00",
    )

    assert payload["dataset_variant"] == "percent10"
    assert payload["selection_metric"] == "qini_auc"
    assert payload["tie_breaker_metric"] == "policy_gain_top20"
    assert payload["best_model_family"] == "s_learner_logistic"
    assert set(payload["models"]) == {"ft_transformer", "s_learner_logistic"}
    assert "generated_at_utc" in payload


def test_summary_schema_and_variant_specific_output_paths(tmp_path: Path) -> None:
    config = load_tuning_config("configs/tuning_full.yaml")
    config.artifacts.tuning_dir = tmp_path / "artifacts" / "tuning" / "full"
    paths = resolve_tuning_output_paths(config)
    studies = {
        "ft_transformer": _completed_study("ft", 0.2, 0.05),
        "s_learner_logistic": _completed_study("s", 0.1, 0.04),
    }
    best_payload = build_best_params_payload(
        config,
        {
            "ft_transformer": {
                "model_family": "ft_transformer",
                "best_value": 0.2,
                "tie_breaker_value": 0.05,
                "best_params": {"embedding_dim": 32, "num_heads": 4},
            },
            "s_learner_logistic": {
                "model_family": "s_learner_logistic",
                "best_value": 0.1,
                "tie_breaker_value": 0.04,
                "best_params": {"C": 1.0, "max_iter": 500},
            },
        },
        generated_at_utc="2026-05-03T00:00:00+00:00",
    )

    summary = build_tuning_summary_payload(
        config,
        studies,
        best_payload,
        paths,
        generated_at_utc="2026-05-03T00:00:00+00:00",
    )

    assert paths["best_params"].as_posix().endswith("artifacts/tuning/full/best_params.json")
    assert summary["dataset_variant"] == "full"
    assert summary["n_trials_requested"]["ft_transformer"] == 8
    assert summary["n_trials_completed"]["ft_transformer"] == 1
    assert summary["best_model_family"] == "ft_transformer"
    assert summary["best_qini_auc"] == 0.2
    assert summary["best_params_path"] == str(paths["best_params"])
    assert summary["trials_path"] == str(paths["trials"])


def test_write_tuning_artifacts_creates_expected_json_and_csv(tmp_path: Path) -> None:
    config = load_tuning_config("configs/tuning.yaml")
    config.artifacts.tuning_dir = tmp_path / "artifacts" / "tuning" / "percent10"
    paths = resolve_tuning_output_paths(config)
    studies = {
        "ft_transformer": _completed_study("ft-write", 0.2, 0.05),
        "s_learner_logistic": _completed_study("s-write", 0.1, 0.04),
    }

    best_payload, summary_payload = write_tuning_artifacts(config, studies, paths)

    assert paths["best_params"].exists()
    assert paths["summary"].exists()
    assert paths["trials"].exists()
    assert json.loads(paths["best_params"].read_text())["models"] == best_payload["models"]
    assert json.loads(paths["summary"].read_text())["trials_path"] == summary_payload["trials_path"]


def test_use_best_params_path_resolution_is_variant_specific() -> None:
    artifact_root = Path("artifacts")
    percent_config = TrainingConfig(data=DataConfig(percent10=True))
    full_config = TrainingConfig(data=DataConfig(percent10=False))

    assert resolve_best_params_path(percent_config, artifact_root) == Path(
        "artifacts/tuning/percent10/best_params.json"
    )
    assert resolve_best_params_path(full_config, artifact_root) == Path(
        "artifacts/tuning/full/best_params.json"
    )
