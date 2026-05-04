from __future__ import annotations

import numpy as np
import torch

from causal_uplift.causal_transformer import (
    CausalFTTransformerTrainer,
    CausalFTTransformerUpliftModel,
)


def test_causal_transformer_forward_shapes_with_propensity() -> None:
    batch_size = 5
    feature_count = 12
    model = CausalFTTransformerUpliftModel(
        num_features=feature_count,
        embedding_dim=32,
        num_layers=2,
        num_heads=4,
        hidden_dim=64,
        use_propensity_head=True,
    )

    outputs = model(torch.randn(batch_size, feature_count))

    assert outputs["control_logit"].shape == (batch_size,)
    assert outputs["treatment_logit"].shape == (batch_size,)
    assert outputs["propensity_logit"].shape == (batch_size,)


def test_causal_transformer_predict_shapes() -> None:
    batch_size = 7
    feature_count = 12
    model = CausalFTTransformerUpliftModel(
        num_features=feature_count,
        embedding_dim=32,
        num_layers=1,
        num_heads=4,
        hidden_dim=64,
        use_propensity_head=False,
    )
    trainer = CausalFTTransformerTrainer(
        model,
        batch_size=3,
        epochs=1,
        log_mlflow_epochs=False,
    )

    prediction = trainer.predict(np.random.default_rng(3).normal(size=(batch_size, feature_count)))

    assert prediction.treatment_probability.shape == (batch_size,)
    assert prediction.control_probability.shape == (batch_size,)
    assert prediction.uplift.shape == (batch_size,)
