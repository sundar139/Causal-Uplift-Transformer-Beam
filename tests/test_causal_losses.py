from __future__ import annotations

import torch
import torch.nn.functional as F

from causal_uplift.causal_losses import (
    compute_positive_class_weights,
    factual_two_head_bce_loss,
    group_balanced_sample_weights,
)


def test_factual_loss_uses_observed_treatment_head() -> None:
    control_logit = torch.tensor([-2.0, -1.0, 5.0, 5.0])
    treatment_logit = torch.tensor([5.0, 5.0, 1.0, 2.0])
    y = torch.tensor([0.0, 1.0, 1.0, 0.0])
    treatment = torch.tensor([0.0, 0.0, 1.0, 1.0])

    loss, components = factual_two_head_bce_loss(
        control_logit,
        treatment_logit,
        y,
        treatment,
        group_balance_weight=0.0,
    )
    expected_logits = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    expected = F.binary_cross_entropy_with_logits(expected_logits, y)

    assert torch.isclose(loss, expected)
    assert components["factual_loss"] >= 0.0


def test_group_balanced_weights_are_finite_and_non_negative() -> None:
    weights = group_balanced_sample_weights(torch.tensor([0.0, 0.0, 0.0, 1.0]))

    assert torch.isfinite(weights).all()
    assert (weights >= 0.0).all()


def test_positive_class_weights_are_finite() -> None:
    weight = compute_positive_class_weights(torch.tensor([0.0, 0.0, 1.0, 0.0]))

    assert torch.isfinite(weight)
    assert float(weight) >= 1.0
