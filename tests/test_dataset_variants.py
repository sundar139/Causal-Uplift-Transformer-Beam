from __future__ import annotations

from pathlib import Path

import pandas as pd

from causal_uplift import data as data_module
from causal_uplift.config import (
    ArtifactConfig,
    DataConfig,
    TrainingConfig,
    TrainingRuntimeConfig,
    TransformerConfig,
    load_training_config,
)
from causal_uplift.data import resolve_materialization_paths
from causal_uplift.train import resolve_training_output_paths


def test_materialization_paths_differ_between_variants() -> None:
    percent10_cfg = load_training_config(Path("configs/training.yaml"))
    full_cfg = load_training_config(Path("configs/training_full.yaml"))

    percent10_paths = resolve_materialization_paths(percent10_cfg)
    full_paths = resolve_materialization_paths(full_cfg)

    assert percent10_paths["raw_path"] != full_paths["raw_path"]
    assert "percent10" in str(percent10_paths["processed_paths"]["train"])
    assert "full" in str(full_paths["processed_paths"]["train"])


def test_profile_artifact_paths_are_variant_specific() -> None:
    percent10_cfg = load_training_config(Path("configs/training.yaml"))
    full_cfg = load_training_config(Path("configs/training_full.yaml"))

    percent10_artifact_dir = resolve_materialization_paths(percent10_cfg)["artifact_dir"]
    full_artifact_dir = resolve_materialization_paths(full_cfg)["artifact_dir"]

    assert "artifacts/data/percent10" in str(percent10_artifact_dir).replace("\\", "/")
    assert "artifacts/data/full" in str(full_artifact_dir).replace("\\", "/")


def test_training_and_report_artifact_paths_are_variant_specific() -> None:
    percent10_cfg = load_training_config(Path("configs/training.yaml"))
    full_cfg = load_training_config(Path("configs/training_full.yaml"))

    percent10_output = resolve_training_output_paths(percent10_cfg, Path("artifacts"))
    full_output = resolve_training_output_paths(full_cfg, Path("artifacts"))

    assert "evaluation/percent10" in str(percent10_output["evaluation_dir"]).replace("\\", "/")
    assert "evaluation/full" in str(full_output["evaluation_dir"]).replace("\\", "/")


def test_no_generated_path_contains_phase() -> None:
    configs = [
        load_training_config(Path("configs/training.yaml")),
        load_training_config(Path("configs/training_full.yaml")),
    ]

    for cfg in configs:
        materialization = resolve_materialization_paths(cfg)
        training_outputs = resolve_training_output_paths(cfg, Path("artifacts"))

        candidates = [
            str(materialization["raw_path"]),
            str(materialization["artifact_dir"]),
            str(training_outputs["evaluation_dir"]),
            str(training_outputs["reports_dir"]),
            str(training_outputs["plots_dir"]),
            str(cfg.artifacts.metrics_name),
            str(cfg.artifacts.predictions_name),
            str(cfg.artifacts.best_model_name),
        ] + [str(path) for path in materialization["processed_paths"].values()]

        for value in candidates:
            assert "phase" not in value.lower()


def test_profile_writes_variant_specific_artifacts(tmp_path: Path, monkeypatch) -> None:
    split_frames = {
        "train": pd.DataFrame(
            {
                "f0": [0.1, 0.2],
                "conversion": [1, 0],
                "treatment": [1, 0],
                "split": ["train", "train"],
            }
        ),
        "validation": pd.DataFrame(
            {
                "f0": [0.3],
                "conversion": [0],
                "treatment": [1],
                "split": ["validation"],
            }
        ),
        "test": pd.DataFrame(
            {
                "f0": [0.4],
                "conversion": [1],
                "treatment": [0],
                "split": ["test"],
            }
        ),
    }

    config = TrainingConfig(
        random_state=42,
        data=DataConfig(percent10=False),
        training=TrainingRuntimeConfig(),
        transformer=TransformerConfig(),
        artifacts=ArtifactConfig(),
        models=["ft_transformer"],
    )

    processed_paths = {
        "train": tmp_path / "processed" / "full" / "train.parquet",
        "validation": tmp_path / "processed" / "full" / "validation.parquet",
        "test": tmp_path / "processed" / "full" / "test.parquet",
    }
    artifact_dir = tmp_path / "artifacts" / "data" / "full"

    monkeypatch.setattr(data_module, "load_training_config", lambda _path: config)
    monkeypatch.setattr(
        data_module,
        "_load_processed_frames_if_available",
        lambda _config: (split_frames, processed_paths),
    )
    monkeypatch.setattr(
        data_module,
        "resolve_materialization_paths",
        lambda _config: {
            "dataset_variant": "full",
            "raw_path": tmp_path / "raw" / "criteo_full.parquet",
            "processed_paths": processed_paths,
            "artifact_dir": artifact_dir,
        },
    )

    result = data_module.profile_dataset("configs/training_full.yaml")

    assert result["dataset_variant"] == "full"
    assert (artifact_dir / "criteo_data_profile.json").exists()
    assert (artifact_dir / "criteo_schema.json").exists()
    assert (artifact_dir / "criteo_sample_preview.csv").exists()
    assert (artifact_dir / "data_manifest.json").exists()
