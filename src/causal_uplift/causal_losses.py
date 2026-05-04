from __future__ import annotations

import torch
import torch.nn.functional as F


def group_balanced_sample_weights(
    treatment: torch.Tensor,
    group_balance_weight: float = 1.0,
) -> torch.Tensor:
    treatment_float = treatment.float()
    treated_mask = treatment_float == 1.0
    control_mask = treatment_float == 0.0
    weights = torch.ones_like(treatment_float)

    treated_count = treated_mask.sum().clamp_min(1).float()
    control_count = control_mask.sum().clamp_min(1).float()
    total = treated_count + control_count
    treated_weight = total / (2.0 * treated_count)
    control_weight = total / (2.0 * control_count)
    balanced = torch.where(treated_mask, treated_weight, control_weight)
    return (1.0 - group_balance_weight) * weights + group_balance_weight * balanced


def compute_positive_class_weights(y: torch.Tensor) -> torch.Tensor:
    y_float = y.float()
    positives = y_float.sum().clamp_min(1.0)
    negatives = (1.0 - y_float).sum().clamp_min(1.0)
    return (negatives / positives).clamp_min(1.0)


def factual_two_head_bce_loss(
    control_logit: torch.Tensor,
    treatment_logit: torch.Tensor,
    y: torch.Tensor,
    treatment: torch.Tensor,
    *,
    group_balance_weight: float = 1.0,
    positive_class_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    y_float = y.float()
    treatment_float = treatment.float()
    observed_logit = torch.where(treatment_float == 1.0, treatment_logit, control_logit)
    per_row_loss = F.binary_cross_entropy_with_logits(
        observed_logit,
        y_float,
        pos_weight=positive_class_weight,
        reduction="none",
    )
    sample_weights = group_balanced_sample_weights(treatment_float, group_balance_weight)
    factual_loss = (per_row_loss * sample_weights).mean()
    treated_loss = per_row_loss[treatment_float == 1.0].mean()
    control_loss = per_row_loss[treatment_float == 0.0].mean()

    if torch.isnan(treated_loss):
        treated_loss = torch.zeros((), device=y.device)
    if torch.isnan(control_loss):
        control_loss = torch.zeros((), device=y.device)

    return factual_loss, {
        "factual_loss": float(factual_loss.detach().cpu().item()),
        "treated_factual_loss": float(treated_loss.detach().cpu().item()),
        "control_factual_loss": float(control_loss.detach().cpu().item()),
    }


def propensity_bce_loss(
    propensity_logit: torch.Tensor,
    treatment: torch.Tensor,
    *,
    group_balance_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    treatment_float = treatment.float()
    per_row_loss = F.binary_cross_entropy_with_logits(
        propensity_logit,
        treatment_float,
        reduction="none",
    )
    sample_weights = group_balanced_sample_weights(treatment_float, group_balance_weight)
    loss = (per_row_loss * sample_weights).mean()
    return loss, {"propensity_loss": float(loss.detach().cpu().item())}
