from __future__ import annotations

from pathlib import Path

from causal_uplift.config import load_training_config


def test_training_config_loads() -> None:
    config_path = Path("configs/training.yaml")
    cfg = load_training_config(config_path)

    assert cfg.data.sample_size >= 0
    assert cfg.data.validation_size > 0
    assert cfg.data.test_size > 0
    assert cfg.data.percent10 is True
    assert "ft_transformer" in cfg.models
    assert cfg.transformer.embedding_dim > 0
    assert cfg.transformer.num_heads > 0


def test_full_training_config_loads() -> None:
    config_path = Path("configs/training_full.yaml")
    cfg = load_training_config(config_path)

    assert cfg.random_state == 42
    assert cfg.data.percent10 is False
    assert cfg.data.sample_size == 0
    assert cfg.data.validation_size == 0.2
    assert cfg.data.test_size == 0.2
    assert cfg.training.batch_size == 4096
    assert cfg.training.max_epochs == 20
    assert cfg.training.early_stopping_patience == 4
    assert cfg.transformer.embedding_dim == 64
    assert cfg.transformer.hidden_dim == 128
    assert cfg.artifacts.best_model_name == "best_transformer_uplift_full.pt"
