from __future__ import annotations

import json
import re
import sys
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
KNOWN_MODEL_NAMES = {
    "s_learner_logistic",
    "t_learner_logistic",
    "two_model_logistic",
    "ft_transformer",
    "causal_ft_transformer",
}
MODEL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
WINDOWS_USER_PATH_PATTERN = re.compile(r"^[A-Za-z]:\\Users\\")


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


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    missing_paths: list[str] = []
    if not BUNDLE_DIR.exists() or not BUNDLE_DIR.is_dir():
        missing_paths.append(str(BUNDLE_DIR))

    for filename in REQUIRED_FILES:
        file_path = BUNDLE_DIR / filename
        if not file_path.exists() or not file_path.is_file():
            missing_paths.append(str(file_path))

    metadata: dict[str, Any] = {}
    if not missing_paths:
        metadata = _load_json(BUNDLE_DIR / "model_metadata.json")

    champion_model = str(metadata.get("champion_model", "")).strip()
    champion_model_valid = champion_model in KNOWN_MODEL_NAMES or bool(
        MODEL_NAME_PATTERN.match(champion_model)
    )

    leaked_paths: list[str] = []
    if metadata:
        for value in _collect_strings(metadata):
            if WINDOWS_USER_PATH_PATTERN.match(value):
                leaked_paths.append(value)

    checks = {
        "bundle_dir_exists": not missing_paths or BUNDLE_DIR.exists(),
        "required_files_present": len(missing_paths) == 0,
        "champion_model_valid": champion_model_valid,
        "metadata_has_no_windows_user_path": len(leaked_paths) == 0,
    }
    ok = all(checks.values())

    summary = {
        "ok": ok,
        "bundle_dir": str(BUNDLE_DIR),
        "champion_model": champion_model,
        "checks": checks,
        "missing_paths": missing_paths,
        "leaked_windows_paths": leaked_paths,
    }
    print(json.dumps(summary, indent=2))

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
