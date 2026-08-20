"""Composite MINAKANUSHI training objectives.

There is no universal next-token loss. August 2026 latent world-model results
(PhyLatent, PSG-JEPA) are used as engineering constraints on the objective,
not as architectural identity:

- physical state grounding (L_state)
- multi-horizon future alignment (L_future)
- counterfactual branch separation (L_action)
- isotropic latent regularizer to prevent collapse (L_representation)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from minakanushi.architecture.config import TrainingConfig
from minakanushi.state.correction import MISSING_CHANNEL, STATE_CHANNEL
from minakanushi.uncertainty.calibration import gaussian_nll


@dataclass
class ObjectiveBreakdown:
    total: Tensor
    terms: dict[str, Tensor]


def isotropic_regularizer(latent: Tensor, mask: Tensor) -> Tensor:
    """Push occupied latents toward zero-mean unit-variance per channel."""
    weights = mask.to(latent.dtype).unsqueeze(-1)
    denom = weights.sum().clamp_min(1.0)
    mean = (latent * weights).sum(dim=(0, 1)) / denom
    centered = (latent - mean) * weights
    var = (centered.pow(2).sum(dim=(0, 1)) / denom).clamp_min(1e-6)
    return mean.pow(2).mean() + (var - 1.0).pow(2).mean()


def counterfactual_separation(pred_a: Tensor, pred_b: Tensor, margin: float) -> Tensor:
    dist = torch.linalg.vector_norm(pred_a - pred_b, dim=-1).mean()
    return torch.relu(margin - dist)


def compute_objectives(
    *,
    pred_xy: Tensor,
    true_xy: Tensor,
    occupied: Tensor,
    pred_next_xy: Tensor,
    true_next_xy: Tensor,
    pred_future_xy: Tensor,
    true_future_xy: Tensor,
    uncertainty: Tensor,
    memory_xy: Tensor,
    memory_true_xy: Tensor,
    memory_mask: Tensor,
    causal_pred: Tensor,
    causal_true: Tensor,
    alt_future_xy: Tensor,
    intra_branch_xy: Tensor,
    latent: Tensor,
    training: TrainingConfig,
    unobserved_mask: Tensor | None = None,
) -> ObjectiveBreakdown:
    """Shapes:
    pred_xy/true_xy:     [B, N, 2]
    occupied:            [B, N]
    pred_future_xy:      [B, H, N, 2]
    uncertainty:         [B, N, U]
    latent:              [B, N, D]
    """
    mask = occupied.to(pred_xy.dtype).unsqueeze(-1)
    denom = mask.sum().clamp_min(1.0)
    l_state = ((pred_xy - true_xy).pow(2) * mask).sum() / denom
    l_temporal = ((pred_next_xy - true_next_xy).pow(2) * mask).sum() / denom
    future_mask = occupied.unsqueeze(1).unsqueeze(-1).to(pred_future_xy.dtype)
    l_future = ((pred_future_xy - true_future_xy).pow(2) * future_mask).sum() / future_mask.sum().clamp_min(1.0)
    error = pred_xy - true_xy
    l_unc_state = gaussian_nll(error, uncertainty, channel=STATE_CHANNEL)
    l_unc_state = (l_unc_state * occupied.to(l_unc_state.dtype)).sum() / occupied.to(l_unc_state.dtype).sum().clamp_min(1.0)
    unobserved = unobserved_mask if unobserved_mask is not None else torch.zeros_like(occupied)
    l_unc_missing = (uncertainty[..., MISSING_CHANNEL] - unobserved.to(uncertainty.dtype)).pow(2)
    l_unc_missing = (l_unc_missing * occupied.to(l_unc_missing.dtype)).sum() / occupied.to(l_unc_missing.dtype).sum().clamp_min(1.0)
    l_uncertainty = l_unc_state + l_unc_missing
    mem_w = memory_mask.to(memory_xy.dtype).unsqueeze(-1)
    l_memory = ((memory_xy - memory_true_xy).pow(2) * mem_w).sum() / mem_w.sum().clamp_min(1.0)
    l_causal = ((causal_pred - causal_true).pow(2) * mask).sum() / denom
    l_action = counterfactual_separation(
        pred_future_xy[:, -1], alt_future_xy[:, -1], training.regularizer.counterfactual_margin
    ) + counterfactual_separation(
        pred_future_xy[:, -1], intra_branch_xy[:, -1], training.regularizer.counterfactual_margin
    )
    l_repr = isotropic_regularizer(latent, occupied)
    lambdas = training.lambdas
    total = (
        lambdas.state * l_state
        + lambdas.temporal * l_temporal
        + lambdas.future * l_future
        + lambdas.uncertainty * l_uncertainty
        + lambdas.causal * l_causal
        + lambdas.memory * l_memory
        + lambdas.action * l_action
        + lambdas.representation * l_repr
    )
    return ObjectiveBreakdown(
        total=total,
        terms={
            "state": l_state,
            "temporal": l_temporal,
            "future": l_future,
            "uncertainty": l_uncertainty,
            "causal": l_causal,
            "memory": l_memory,
            "action": l_action,
            "representation": l_repr,
        },
    )
