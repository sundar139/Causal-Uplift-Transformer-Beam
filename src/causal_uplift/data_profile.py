from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def build_schema_payload(
    frame: pd.DataFrame,
    target_column: str,
    treatment_column: str,
) -> dict[str, object]:
    feature_columns = [
        column
        for column in frame.columns
        if column not in {target_column, treatment_column, "split"}
    ]

    return {
        "feature_names": feature_columns,
        "target_column": target_column,
        "treatment_column": treatment_column,
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "number_of_features": len(feature_columns),
    }


def _split_rates(
    split_frames: dict[str, pd.DataFrame],
    column_name: str,
) -> dict[str, float]:
    rates: dict[str, float] = {}
    for split_name, split_frame in split_frames.items():
        if split_frame.empty:
            rates[split_name] = 0.0
        else:
            rates[split_name] = float(split_frame[column_name].astype(float).mean())
    return rates


def build_profile_payload(
    split_frames: dict[str, pd.DataFrame],
    target_column: str,
    treatment_column: str,
) -> dict[str, object]:
    merged = pd.concat(split_frames.values(), ignore_index=True)
    feature_columns = [
        column
        for column in merged.columns
        if column not in {target_column, treatment_column, "split"}
    ]

    numeric_summary = (
        merged[feature_columns]
        .select_dtypes(include=["number", "bool"])
        .describe()
        .transpose()
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    return {
        "total_rows": int(len(merged)),
        "split_row_counts": {
            split_name: int(len(split_frame)) for split_name, split_frame in split_frames.items()
        },
        "conversion_rate_per_split": _split_rates(split_frames, target_column),
        "treatment_rate_per_split": _split_rates(split_frames, treatment_column),
        "missing_value_counts": {
            column: int(value) for column, value in merged.isna().sum().items()
        },
        "basic_numeric_summary": numeric_summary.to_dict(orient="records"),
    }


def build_manifest_payload(
    *,
    dataset_name: str,
    source: str,
    loader: str,
    target_column: str,
    treatment_column: str,
    feature_columns: list[str],
    split_frames: dict[str, pd.DataFrame],
    random_state: int,
    local_raw_path: Path,
    local_processed_paths: dict[str, Path],
) -> dict[str, object]:
    return {
        "dataset_name": dataset_name,
        "source": source,
        "loader": loader,
        "target_column": target_column,
        "treatment_column": treatment_column,
        "feature_columns": feature_columns,
        "row_counts": {
            split_name: int(len(split_frame)) for split_name, split_frame in split_frames.items()
        },
        "outcome_rate": _split_rates(split_frames, target_column),
        "treatment_rate": _split_rates(split_frames, treatment_column),
        "random_state": random_state,
        "generated_at_utc": utc_timestamp(),
        "local_raw_path": str(local_raw_path),
        "local_processed_paths": {
            split_name: str(path) for split_name, path in local_processed_paths.items()
        },
        "git_tracking_policy": (
            "Raw and processed parquet files are git-ignored. "
            "Lightweight artifacts in artifacts/data are tracked for lineage."
        ),
    }


def build_sample_preview(frame: pd.DataFrame, max_rows: int = 100) -> pd.DataFrame:
    return frame.head(max_rows).copy()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
