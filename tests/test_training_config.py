from __future__ import annotations

from pathlib import Path

from causal_uplift.config import load_training_config


def test_training_config_loads() -> None:
    config_path = Path("configs/training.yaml")
    cfg = load_training_config(config_path)

    assert cfg.sample_size >= 0
    assert cfg.split.validation_size > 0
    assert cfg.split.test_size > 0
    assert "ft_transformer" in cfg.models
    assert cfg.transformer.d_token > 0
    assert cfg.transformer.num_heads > 0
