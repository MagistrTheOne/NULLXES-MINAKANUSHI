"""Composite MINAKANUSHI training objectives.

There is no universal next-token loss. August 2026 latent world-model results
(PhyLatent, PSG-JEPA) are used as engineering constraints on the objective,
not as architectural identity:

- physical state grounding (L_state) — auxiliary, not the belief definition
- belief NLL + existence (L_belief)
- belief revision vs evidence (L_revision): detection, direction, calibration
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
from minakanushi.training.revision import revision_losses
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


def belief_nll(mean: Tensor, std: Tensor, true: Tensor, mask: Tensor) -> Tensor:
    """Gaussian NLL of GT under (mean, std). mask [B, N] bool/float."""
    sigma = std.clamp_min(1e-3)
    nll = 0.5 * (((mean - true) / sigma).pow(2) + 2.0 * torch.log(sigma)).sum(dim=-1)
    weights = mask.to(nll.dtype)
    return (nll * weights).sum() / weights.sum().clamp_min(1.0)


def existence_bce(existence: Tensor, true_present: Tensor, occupied: Tensor) -> Tensor:
    """BCE of existence vs whether the hypothesized slot was actually present."""
    p = existence.clamp(1e-4, 1.0 - 1e-4)
    t = true_present.to(p.dtype)
    bce = -(t * torch.log(p) + (1.0 - t) * torch.log(1.0 - p))
    weights = occupied.to(p.dtype)
    return (bce * weights).sum() / weights.sum().clamp_min(1.0)


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
    xy_std: Tensor | None = None,
    existence: Tensor | None = None,
    true_present: Tensor | None = None,
    hypothesized: Tensor | None = None,
    before_xy: Tensor | None = None,
    after_xy: Tensor | None = None,
    after_vel: Tensor | None = None,
    evidence_xy: Tensor | None = None,
    evidence_vel: Tensor | None = None,
    should_revise: Tensor | None = None,
    has_evidence: Tensor | None = None,
    occupied_before: Tensor | None = None,
    entity_id: Tensor | None = None,
) -> ObjectiveBreakdown:
    """Shapes:
    pred_xy/true_xy:     [B, N, 2]
    occupied:            [B, N]
    pred_future_xy:      [B, H, N, 2]
    uncertainty:         [B, N, U]
    latent:              [B, N, D]
    xy_std:              [B, N, 2]
    existence:           [B, N]
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
    if xy_std is None:
        l_belief_nll = pred_xy.new_zeros(())
    else:
        l_belief_nll = belief_nll(pred_xy, xy_std, true_xy, occupied)
    if existence is None or true_present is None:
        l_exist = pred_xy.new_zeros(())
    else:
        exist_mask = hypothesized if hypothesized is not None else occupied
        l_exist = existence_bce(existence, true_present, exist_mask)
    l_belief = l_belief_nll + l_exist
    zero = pred_xy.new_zeros(())
    rev_parts: dict[str, Tensor] = {
        "revision_detection": zero,
        "revision_direction": zero,
        "revision_calibration": zero,
        "revision_false": zero,
    }
    if (
        before_xy is None
        or evidence_xy is None
        or should_revise is None
        or has_evidence is None
        or occupied_before is None
        or entity_id is None
    ):
        l_revision = zero
    else:
        after = after_xy if after_xy is not None else pred_xy
        vel = after_vel if after_vel is not None else torch.zeros_like(after)
        ev_vel = evidence_vel if evidence_vel is not None else torch.zeros_like(after)
        l_revision, rev_parts = revision_losses(
            before_xy=before_xy,
            after_xy=after,
            after_vel=vel,
            evidence_xy=evidence_xy,
            evidence_vel=ev_vel,
            should_revise=should_revise,
            has_evidence=has_evidence,
            occupied_before=occupied_before,
            entity_id=entity_id,
        )
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
        + lambdas.belief * l_belief
        + lambdas.revision * l_revision
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
            "belief": l_belief,
            "revision": l_revision,
            **rev_parts,
        },
    )
