from __future__ import annotations

from pathlib import Path

from causal_uplift.config import AppConfig


def test_smoke_imports() -> None:
    import causal_uplift.baselines  # noqa: F401
    import causal_uplift.data  # noqa: F401
    import causal_uplift.data_profile  # noqa: F401
    import causal_uplift.evaluate  # noqa: F401
    import causal_uplift.mlflow_utils  # noqa: F401
    import causal_uplift.plots  # noqa: F401
    import causal_uplift.preprocessing  # noqa: F401
    import causal_uplift.reporting  # noqa: F401
    import causal_uplift.serve  # noqa: F401
    import causal_uplift.train  # noqa: F401
    import causal_uplift.transformer  # noqa: F401


def test_app_config_creates_directories(tmp_path: Path, monkeypatch) -> None:
    model_dir = tmp_path / "models"
    artifact_dir = tmp_path / "artifacts"

    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setenv("ARTIFACT_DIR", str(artifact_dir))

    config = AppConfig.from_env()

    assert config.model_dir.exists()
    assert config.artifact_dir.exists()
