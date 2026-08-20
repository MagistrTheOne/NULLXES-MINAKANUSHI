"""Belief-revision training signal.

CorrectionEvent already exists on WorldState. This module turns
(old belief, new evidence, DWC after) into L_revision so the residual
cannot undo a justified update for free.

Not a constructor rewrite. Not a DWC architecture change.
"""

from __future__ import annotations

import torch
from torch import Tensor

from minakanushi.architecture.mina_unit import MinaUnitBatch
from minakanushi.state.correction import REVISION_MAGNITUDE
from minakanushi.state.world import WorldState

AGENT_ENTITY_ID = 1
MOVE_DETECT = 0.05


def evidence_for_slots(world: WorldState, units: MinaUnitBatch) -> tuple[Tensor, Tensor, Tensor]:
    """Align observation evidence onto world slots by entity_id.

    evidence_xy:  [B, N, 2]
    evidence_vel: [B, N, 2]
    has_evidence: [B, N] bool
    """
    evidence_xy = torch.zeros_like(world.entity_xy)
    evidence_vel = torch.zeros_like(world.entity_vel)
    has_evidence = torch.zeros_like(world.occupied)
    for b in range(world.entity_id.shape[0]):
        for i in range(units.mask.shape[1]):
            if not bool(units.mask[b, i]):
                continue
            if not bool(units.spatial_valid[b, i]):
                continue
            eid = int(units.entity_id[b, i].item())
            if eid <= 0:
                continue
            hits = ((world.entity_id[b] == eid) & world.occupied[b]).nonzero(as_tuple=False)
            if hits.numel() == 0:
                continue
            slot = int(hits[0].item())
            evidence_xy[b, slot] = units.spatial_position[b, i, :2]
            evidence_vel[b, slot] = units.velocity[b, i]
            has_evidence[b, slot] = True
    return evidence_xy, evidence_vel, has_evidence


def should_revise_mask(
    before_xy: Tensor,
    evidence_xy: Tensor,
    has_evidence: Tensor,
    occupied_before: Tensor,
    entity_id: Tensor,
    magnitude: float = REVISION_MAGNITUDE,
) -> Tensor:
    """True where an existing non-self hypothesis disagrees with evidence."""
    delta = torch.linalg.vector_norm(before_xy - evidence_xy, dim=-1)
    not_self = entity_id != AGENT_ENTITY_ID
    return has_evidence & occupied_before & not_self & (delta >= magnitude)


def revision_losses(
    *,
    before_xy: Tensor,
    after_xy: Tensor,
    after_vel: Tensor,
    evidence_xy: Tensor,
    evidence_vel: Tensor,
    should_revise: Tensor,
    has_evidence: Tensor,
    occupied_before: Tensor,
    entity_id: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Split L_revision: detection + direction + calibration (+ false revision).

    before is detached prior belief. after is DWC output (has gradient).
    """
    before_d = torch.linalg.vector_norm(before_xy - evidence_xy, dim=-1)
    after_d = torch.linalg.vector_norm(after_xy - evidence_xy, dim=-1)
    moved = torch.linalg.vector_norm(after_xy - before_xy, dim=-1)
    w = should_revise.to(after_xy.dtype)
    w_sum = w.sum().clamp_min(1.0)
    detection = (w * torch.exp(-moved)).sum() / w_sum
    direction = (w * torch.relu(after_d - before_d)).sum() / w_sum
    mag = (w * after_d.pow(2)).sum() / w_sum
    vel_err = (w.unsqueeze(-1) * (after_vel - evidence_vel).pow(2)).sum() / (w_sum * 2.0)
    calibration = mag + vel_err
    not_self = entity_id != AGENT_ENTITY_ID
    false_w = (has_evidence & occupied_before & ~should_revise & not_self).to(after_xy.dtype)
    false_den = false_w.sum().clamp_min(1.0)
    false_revision = (false_w * torch.relu(moved - REVISION_MAGNITUDE)).sum() / false_den
    total = detection + direction + calibration + 0.5 * false_revision
    return total, {
        "revision_detection": detection,
        "revision_direction": direction,
        "revision_calibration": calibration,
        "revision_false": false_revision,
    }


def revision_metrics(
    *,
    before_xy: Tensor,
    after_xy: Tensor,
    evidence_xy: Tensor,
    should_revise: Tensor,
    has_evidence: Tensor,
    occupied_before: Tensor,
    entity_id: Tensor,
) -> dict[str, float]:
    before_d = torch.linalg.vector_norm(before_xy - evidence_xy, dim=-1)
    after_d = torch.linalg.vector_norm(after_xy - evidence_xy, dim=-1)
    moved = torch.linalg.vector_norm(after_xy - before_xy, dim=-1)
    toward = after_d < before_d
    n_need = int(should_revise.sum().item())
    if n_need == 0:
        detected = 0.0
        direction = 0.0
        mag_err = 0.0
        latency = -1.0
    else:
        detected = float((should_revise & (moved > MOVE_DETECT)).sum().item() / n_need)
        direction = float((should_revise & toward).sum().item() / n_need)
        mag_err = float(after_d[should_revise].mean().item())
        latency = 0.0 if direction > 0.0 else -1.0
    not_self = entity_id != AGENT_ENTITY_ID
    stable = has_evidence & occupied_before & ~should_revise & not_self
    n_stable = int(stable.sum().item())
    false_rate = float((stable & (moved > REVISION_MAGNITUDE)).sum().item() / n_stable) if n_stable else 0.0
    return {
        "revision_detected": detected,
        "revision_direction_accuracy": direction,
        "revision_magnitude_error": mag_err,
        "revision_latency": latency,
        "false_revision_rate": false_rate,
        "belief_revision_accuracy": direction,
    }
