from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

BUNDLE_DIR = Path("models/production")
REQUIRED_FILES = (
    "champion_model.joblib",
    "preprocessor.joblib",
    "model_metadata.json",
    "feature_schema.json",
    "example_request.json",
    "prediction_contract.json",
)
WINDOWS_USER_PATH_PATTERN = re.compile(r"^[A-Za-z]:\\Users\\")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _collect_strings(payload: Any) -> list[str]:
    values: list[str] = []
    if isinstance(payload, str):
        values.append(payload)
    elif isinstance(payload, dict):
        for value in payload.values():
            values.extend(_collect_strings(value))
    elif isinstance(payload, list):
        for value in payload:
            values.extend(_collect_strings(value))
    return values


def test_required_production_bundle_files_exist() -> None:
    assert BUNDLE_DIR.is_dir(), "models/production must exist"
    for filename in REQUIRED_FILES:
        assert (BUNDLE_DIR / filename).is_file(), f"Missing required bundle file: {filename}"


def test_metadata_has_no_local_windows_absolute_path_leakage() -> None:
    metadata = _read_json(BUNDLE_DIR / "model_metadata.json")
    leaked = [
        value for value in _collect_strings(metadata) if WINDOWS_USER_PATH_PATTERN.match(value)
    ]

    assert leaked == []


def test_feature_schema_contract() -> None:
    schema = _read_json(BUNDLE_DIR / "feature_schema.json")

    assert isinstance(schema.get("required_columns"), list)
    assert isinstance(schema.get("feature_count"), int)
    assert schema["feature_count"] == len(schema["required_columns"])


def test_example_request_has_12_features() -> None:
    example_request = _read_json(BUNDLE_DIR / "example_request.json")
    features = example_request.get("features")

    assert isinstance(features, dict)
    assert len(features) == 12
