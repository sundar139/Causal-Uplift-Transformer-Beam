from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from causal_uplift.data_profile import (
    build_manifest_payload,
    build_profile_payload,
    build_sample_preview,
    build_schema_payload,
)


def _synthetic_split_frames(rows_per_split: int = 60) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(99)
    frames: dict[str, pd.DataFrame] = {}
    for split_name in ["train", "validation", "test"]:
        frames[split_name] = pd.DataFrame(
            {
                "f0": rng.normal(size=rows_per_split),
                "f1": rng.normal(size=rows_per_split),
                "conversion": rng.integers(0, 2, size=rows_per_split),
                "treatment": rng.integers(0, 2, size=rows_per_split),
                "split": split_name,
            }
        )
    return frames


def test_schema_and_manifest_generation_contract() -> None:
    frames = _synthetic_split_frames()
    merged = pd.concat(frames.values(), ignore_index=True)

    schema = build_schema_payload(
        frame=merged,
        target_column="conversion",
        treatment_column="treatment",
    )
    assert schema["target_column"] == "conversion"
    assert schema["treatment_column"] == "treatment"
    assert schema["number_of_features"] == 2
    assert schema["feature_names"] == ["f0", "f1"]

    manifest = build_manifest_payload(
        dataset_name="Criteo Uplift Prediction (percent10)",
        source="scikit-uplift",
        loader="sklift.datasets.fetch_criteo",
        target_column="conversion",
        treatment_column="treatment",
        feature_columns=["f0", "f1"],
        split_frames=frames,
        random_state=42,
        local_raw_path=Path("data/raw/criteo_percent10.parquet"),
        local_processed_paths={
            "train": Path("data/processed/train.parquet"),
            "validation": Path("data/processed/validation.parquet"),
            "test": Path("data/processed/test.parquet"),
        },
    )
    assert manifest["dataset_name"] == "Criteo Uplift Prediction (percent10)"
    assert set(manifest["row_counts"].keys()) == {"train", "validation", "test"}
    assert "generated_at_utc" in manifest


def test_profile_payload_and_sample_preview_limit() -> None:
    frames = _synthetic_split_frames(rows_per_split=120)
    profile = build_profile_payload(
        split_frames=frames,
        target_column="conversion",
        treatment_column="treatment",
    )

    assert profile["total_rows"] == 360
    assert set(profile["split_row_counts"].keys()) == {"train", "validation", "test"}
    assert "missing_value_counts" in profile
    assert isinstance(profile["basic_numeric_summary"], list)

    merged = pd.concat(frames.values(), ignore_index=True)
    preview = build_sample_preview(merged, max_rows=100)
    assert len(preview) <= 100
