from __future__ import annotations

import torch

from causal_uplift.transformer import FTTransformerUpliftModel


def test_transformer_forward_shape() -> None:
    model = FTTransformerUpliftModel(
        num_features=13,
        d_token=32,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
    )
    batch = torch.randn(8, 13, dtype=torch.float32)
    logits = model(batch)
    assert logits.shape == (8,)
