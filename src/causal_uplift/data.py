from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklift.datasets import fetch_criteo

from causal_uplift.config import AppConfig, TrainingConfig, load_training_config
from causal_uplift.data_profile import (
    build_manifest_payload,
    build_profile_payload,
    build_sample_preview,
    build_schema_payload,
    write_json,
)

TARGET_COLUMN = "conversion"
TREATMENT_COLUMN = "treatment"
PERCENT10_DATASET_NAME = "Criteo Uplift Prediction (percent10)"
FULL_DATASET_NAME = "Criteo Uplift Prediction (full)"
DATASET_SOURCE = "scikit-uplift"
DATASET_LOADER = "sklift.datasets.fetch_criteo"


@dataclass(slots=True)
class CriteoDataset:
    features: pd.DataFrame
    outcomes: pd.Series
    treatment: pd.Series


@dataclass(slots=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    treatment_train: pd.Series
    treatment_validation: pd.Series
    treatment_test: pd.Series


@dataclass(slots=True)
class CriteoSample:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    treatment_train: pd.Series
    treatment_test: pd.Series


def _clean_numeric_features(features: pd.DataFrame) -> pd.DataFrame:
    clean = features.copy()
    numeric_cols = clean.select_dtypes(include=["number", "bool"]).columns
    clean[numeric_cols] = clean[numeric_cols].apply(pd.to_numeric, errors="coerce")
    medians = clean[numeric_cols].median(numeric_only=True)
    clean[numeric_cols] = clean[numeric_cols].fillna(medians)
    return clean


def load_criteo_dataset(
    sample_size: int = 0,
    *,
    percent10: bool = True,
    target_col: str = TARGET_COLUMN,
    treatment_col: str = TREATMENT_COLUMN,
) -> CriteoDataset:
    if not percent10:
        print(
            "WARNING: Loading full Criteo dataset (percent10=false). "
            "This may require significant disk, memory, and training time."
        )

    X, y, treatment = fetch_criteo(
        target_col=target_col,
        treatment_col=treatment_col,
        return_X_y_t=True,
        percent10=percent10,
    )

    X_df = _clean_numeric_features(pd.DataFrame(X).reset_index(drop=True))
    y_series = pd.Series(y).reset_index(drop=True).astype(int)
    treatment_series = pd.Series(treatment).reset_index(drop=True).astype(int)

    if sample_size > 0:
        max_rows = min(sample_size, len(X_df))
        X_df = X_df.iloc[:max_rows].copy()
        y_series = y_series.iloc[:max_rows].copy()
        treatment_series = treatment_series.iloc[:max_rows].copy()

    return CriteoDataset(features=X_df, outcomes=y_series, treatment=treatment_series)


def create_train_validation_test_split(
    dataset: CriteoDataset,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> DatasetSplit:
    if validation_size <= 0 or test_size <= 0 or (validation_size + test_size) >= 1:
        raise ValueError("validation_size and test_size must be positive and sum to < 1")

    strata = dataset.outcomes.astype(str) + "_" + dataset.treatment.astype(str)
    holdout_size = validation_size + test_size

    X_train, X_holdout, y_train, y_holdout, t_train, t_holdout = train_test_split(
        dataset.features,
        dataset.outcomes,
        dataset.treatment,
        test_size=holdout_size,
        random_state=random_state,
        stratify=strata,
    )

    holdout_strata = y_holdout.astype(str) + "_" + t_holdout.astype(str)
    relative_test_size = test_size / holdout_size

    X_validation, X_test, y_validation, y_test, t_validation, t_test = train_test_split(
        X_holdout,
        y_holdout,
        t_holdout,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=holdout_strata,
    )

    return DatasetSplit(
        X_train=X_train.reset_index(drop=True),
        X_validation=X_validation.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_validation=y_validation.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        treatment_train=t_train.reset_index(drop=True),
        treatment_validation=t_validation.reset_index(drop=True),
        treatment_test=t_test.reset_index(drop=True),
    )


def load_criteo_sample(
    sample_size: int = 10_000,
    test_size: float = 0.2,
    random_state: int = 42,
) -> CriteoSample:
    dataset = load_criteo_dataset(sample_size=sample_size, percent10=True)
    X_df = dataset.features
    y_series = dataset.outcomes
    treatment_series = dataset.treatment

    strata = y_series.astype(str) + "_" + treatment_series.astype(str)

    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X_df,
        y_series,
        treatment_series,
        test_size=test_size,
        random_state=random_state,
        stratify=strata,
    )

    return CriteoSample(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        treatment_train=t_train.reset_index(drop=True),
        treatment_test=t_test.reset_index(drop=True),
    )


def _join_split_frame(
    X: pd.DataFrame,
    y: pd.Series,
    treatment: pd.Series,
    split_name: str,
) -> pd.DataFrame:
    joined = X.copy().reset_index(drop=True)
    joined[TARGET_COLUMN] = y.reset_index(drop=True).astype(int)
    joined[TREATMENT_COLUMN] = treatment.reset_index(drop=True).astype(int)
    joined["split"] = split_name
    return joined


def build_processed_split_frames(split: DatasetSplit) -> dict[str, pd.DataFrame]:
    return {
        "train": _join_split_frame(
            X=split.X_train,
            y=split.y_train,
            treatment=split.treatment_train,
            split_name="train",
        ),
        "validation": _join_split_frame(
            X=split.X_validation,
            y=split.y_validation,
            treatment=split.treatment_validation,
            split_name="validation",
        ),
        "test": _join_split_frame(
            X=split.X_test,
            y=split.y_test,
            treatment=split.treatment_test,
            split_name="test",
        ),
    }


def dataset_variant_from_config(config: TrainingConfig) -> str:
    return config.data.dataset_variant


def resolve_materialization_paths(config: TrainingConfig) -> dict[str, object]:
    variant = dataset_variant_from_config(config)
    raw_name = "criteo_percent10.parquet" if config.data.percent10 else "criteo_full.parquet"
    processed_base = Path("data/processed") / variant
    processed_paths = {
        "train": processed_base / "train.parquet",
        "validation": processed_base / "validation.parquet",
        "test": processed_base / "test.parquet",
    }
    return {
        "dataset_variant": variant,
        "raw_path": Path("data/raw") / raw_name,
        "processed_paths": processed_paths,
        "artifact_dir": Path("artifacts/data") / variant,
    }


def _validate_full_vs_percent10_row_counts(full_row_count: int) -> None:
    percent10_manifest = Path("artifacts/data/percent10/data_manifest.json")
    full_manifest = Path("artifacts/data/full/data_manifest.json")
    if not (percent10_manifest.exists() and full_manifest.exists()):
        return

    with percent10_manifest.open("r", encoding="utf-8") as handle:
        percent10_payload = json.load(handle)
    baseline_count = int(sum(percent10_payload.get("row_counts", {}).values()))
    if full_row_count <= baseline_count:
        raise ValueError(
            "Full dataset materialization produced a row count that is not larger than percent10. "
            "Check configuration and dataset source."
        )


def _load_split_from_config(config_path: str | Path) -> tuple[DatasetSplit, CriteoDataset, int]:
    training_config = load_training_config(config_path)

    dataset = load_criteo_dataset(
        sample_size=training_config.data.sample_size,
        percent10=training_config.data.percent10,
        target_col=training_config.data.target_col,
        treatment_col=training_config.data.treatment_col,
    )
    split = create_train_validation_test_split(
        dataset,
        validation_size=training_config.data.validation_size,
        test_size=training_config.data.test_size,
        random_state=training_config.random_state,
    )
    return split, dataset, training_config.random_state


def materialize_dataset(config_path: str | Path) -> dict[str, object]:
    training_config = load_training_config(config_path)
    split, dataset, random_state = _load_split_from_config(config_path)
    split_frames = build_processed_split_frames(split)
    paths = resolve_materialization_paths(training_config)
    raw_path: Path = paths["raw_path"]
    processed_paths: dict[str, Path] = paths["processed_paths"]

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    for path in processed_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    raw_frame = dataset.features.copy()
    raw_frame[TARGET_COLUMN] = dataset.outcomes
    raw_frame[TREATMENT_COLUMN] = dataset.treatment
    raw_frame.to_parquet(raw_path, index=False)

    for split_name, split_frame in split_frames.items():
        split_frame.to_parquet(processed_paths[split_name], index=False)

    total_rows = int(sum(len(frame) for frame in split_frames.values()))
    if not training_config.data.percent10:
        _validate_full_vs_percent10_row_counts(total_rows)

    dataset_name = PERCENT10_DATASET_NAME if training_config.data.percent10 else FULL_DATASET_NAME

    summary = {
        "dataset": dataset_name,
        "dataset_variant": paths["dataset_variant"],
        "percent10": training_config.data.percent10,
        "random_state": random_state,
        "raw_path": str(raw_path),
        "processed_paths": {name: str(path) for name, path in processed_paths.items()},
        "row_counts": {name: int(len(frame)) for name, frame in split_frames.items()},
    }
    print(json.dumps(summary, indent=2))
    return summary


def _load_processed_frames_if_available(
    config: TrainingConfig,
) -> tuple[dict[str, pd.DataFrame] | None, dict[str, Path]]:
    processed_paths = resolve_materialization_paths(config)["processed_paths"]
    if all(path.exists() for path in processed_paths.values()):
        return (
            {split_name: pd.read_parquet(path) for split_name, path in processed_paths.items()},
            processed_paths,
        )
    return None, processed_paths


def profile_dataset(config_path: str | Path) -> dict[str, object]:
    _ = AppConfig.from_env()
    training_config = load_training_config(config_path)
    paths = resolve_materialization_paths(training_config)
    loaded_frames, processed_paths = _load_processed_frames_if_available(training_config)

    if loaded_frames is None:
        split, dataset, random_state = _load_split_from_config(config_path)
        split_frames = build_processed_split_frames(split)
        raw_path: Path = paths["raw_path"]
        feature_columns = list(dataset.features.columns)
    else:
        split_frames = loaded_frames
        raw_path = paths["raw_path"]
        random_state = training_config.random_state
        train_columns = split_frames["train"].columns
        feature_columns = [
            column
            for column in train_columns
            if column not in {TARGET_COLUMN, TREATMENT_COLUMN, "split"}
        ]

    merged = pd.concat(split_frames.values(), ignore_index=True)
    schema_payload = build_schema_payload(
        frame=merged,
        target_column=TARGET_COLUMN,
        treatment_column=TREATMENT_COLUMN,
    )
    profile_payload = build_profile_payload(
        split_frames=split_frames,
        target_column=TARGET_COLUMN,
        treatment_column=TREATMENT_COLUMN,
    )
    manifest_payload = build_manifest_payload(
        dataset_name=(
            PERCENT10_DATASET_NAME if training_config.data.percent10 else FULL_DATASET_NAME
        ),
        source=DATASET_SOURCE,
        loader=f"{DATASET_LOADER}(percent10={training_config.data.percent10})",
        target_column=training_config.data.target_col,
        treatment_column=training_config.data.treatment_col,
        feature_columns=feature_columns,
        split_frames=split_frames,
        random_state=random_state,
        local_raw_path=raw_path,
        local_processed_paths=processed_paths,
    )

    artifact_dir: Path = paths["artifact_dir"]
    profile_path = artifact_dir / "criteo_data_profile.json"
    schema_path = artifact_dir / "criteo_schema.json"
    preview_path = artifact_dir / "criteo_sample_preview.csv"
    manifest_path = artifact_dir / "data_manifest.json"

    write_json(profile_path, profile_payload)
    write_json(schema_path, schema_payload)
    write_json(manifest_path, manifest_payload)
    build_sample_preview(merged, max_rows=100).to_csv(preview_path, index=False)

    summary = {
        "dataset_variant": paths["dataset_variant"],
        "profile_artifact": str(profile_path),
        "schema_artifact": str(schema_path),
        "sample_preview_artifact": str(preview_path),
        "manifest_artifact": str(manifest_path),
    }
    print(json.dumps(summary, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Causal uplift data CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize_parser = subparsers.add_parser(
        "materialize",
        help="Materialize Criteo percent10 raw and split parquet datasets",
    )
    materialize_parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to training configuration",
    )

    profile_parser = subparsers.add_parser(
        "profile",
        help="Generate data profile and lineage artifacts",
    )
    profile_parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to training configuration",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "materialize":
        materialize_dataset(config_path=args.config)
        return

    if args.command == "profile":
        profile_dataset(config_path=args.config)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
