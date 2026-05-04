from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from causal_uplift import bundle
from causal_uplift.preprocessing import NumericFeaturePreprocessor


class FakeChampion:
    pass


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fake_report_payload(model_name: str = "s_learner_logistic") -> dict[str, object]:
    return {
        "best_model": model_name,
        "selection_metric": "qini_auc",
        "qini_auc": 0.2,
        "uplift_auc": 0.1,
        "policy_gain_top10": 0.03,
        "policy_gain_top20": 0.02,
        "policy_gain_top30": 0.01,
        "treatment_response_auc": 0.8,
        "source_predictions_path": "artifacts/evaluation/full/predictions.csv",
        "source_metrics_path": "artifacts/evaluation/full/metrics.json",
    }


def _write_training_config(path: Path) -> None:
    path.write_text(
        """
random_state: 42
data:
  sample_size: 0
  percent10: false
  test_size: 0.2
  validation_size: 0.2
  target_col: conversion
  treatment_col: treatment
artifacts:
  best_model_name: best_transformer_uplift_full.pt
""".strip(),
        encoding="utf-8",
    )


def test_bundle_creation_selects_champion_and_writes_metadata(tmp_path: Path, monkeypatch) -> None:
    artifact_root = tmp_path / "artifacts"
    model_root = tmp_path / "models"
    config_path = tmp_path / "training_full.yaml"
    _write_training_config(config_path)
    _write_json(
        artifact_root / "reports" / "full" / "best_model_summary.json",
        _fake_report_payload(),
    )
    model_root.mkdir(parents=True)
    joblib.dump(FakeChampion(), model_root / "s_learner_logistic.joblib")
    monkeypatch.setenv("ARTIFACT_DIR", str(artifact_root))
    monkeypatch.setenv("MODEL_DIR", str(model_root))

    preprocessor = NumericFeaturePreprocessor().fit(pd.DataFrame({"f0": [1.0], "f1": [2.0]}))
    monkeypatch.setattr(bundle, "fit_preprocessor_for_bundle", lambda _config: preprocessor)

    summary = bundle.build_inference_bundle(config_path, tmp_path / "production")
    metadata = json.loads(Path(summary["metadata_path"]).read_text(encoding="utf-8"))

    assert summary["champion_model"] == "s_learner_logistic"
    assert metadata["champion_model"] == "s_learner_logistic"
    assert Path(summary["feature_schema_path"]).exists()
    assert Path(summary["example_request_path"]).exists()
    assert Path(summary["prediction_contract_path"]).exists()


def test_bundle_metadata_has_no_absolute_local_paths(tmp_path: Path, monkeypatch) -> None:
    artifact_root = tmp_path / "artifacts"
    model_root = tmp_path / "models"
    config_path = tmp_path / "training_full.yaml"
    _write_training_config(config_path)
    report_payload = _fake_report_payload()
    report_payload["source_predictions_path"] = str(tmp_path / "predictions.csv")
    report_payload["source_metrics_path"] = str(tmp_path / "metrics.json")
    _write_json(artifact_root / "reports" / "full" / "best_model_summary.json", report_payload)
    model_root.mkdir(parents=True)
    joblib.dump(FakeChampion(), model_root / "s_learner_logistic.joblib")
    monkeypatch.setenv("ARTIFACT_DIR", str(artifact_root))
    monkeypatch.setenv("MODEL_DIR", str(model_root))
    monkeypatch.setattr(
        bundle,
        "fit_preprocessor_for_bundle",
        lambda _config: NumericFeaturePreprocessor().fit(pd.DataFrame({"f0": [1.0]})),
    )

    summary = bundle.build_inference_bundle(config_path, tmp_path / "production")
    metadata = json.loads(Path(summary["metadata_path"]).read_text(encoding="utf-8"))

    path_fields = [
        "source_report_path",
        "source_model_path",
        "source_predictions_path",
        "source_metrics_path",
    ]
    assert all(not Path(str(metadata[field])).is_absolute() for field in path_fields)


def test_missing_champion_artifact_fails_clearly(tmp_path: Path, monkeypatch) -> None:
    artifact_root = tmp_path / "artifacts"
    model_root = tmp_path / "models"
    config_path = tmp_path / "training_full.yaml"
    _write_training_config(config_path)
    _write_json(
        artifact_root / "reports" / "full" / "best_model_summary.json",
        _fake_report_payload(),
    )
    model_root.mkdir(parents=True)
    monkeypatch.setenv("ARTIFACT_DIR", str(artifact_root))
    monkeypatch.setenv("MODEL_DIR", str(model_root))

    with pytest.raises(FileNotFoundError, match="Selected champion artifact is missing"):
        bundle.build_inference_bundle(config_path, tmp_path / "production")
