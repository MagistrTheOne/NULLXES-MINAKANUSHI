"""Belief revision — evidence updates current belief. This is not memory.

Memory is what was.
Belief is what is probable now.
Future is what may be.

A CorrectionEvent is recorded when new evidence revises a prior hypothesis,
not when a first observation creates a slot.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

RELIABILITY_TAU = 1.0
REVISION_MAGNITUDE = 0.25
TRACKING_GAIN_FLOOR = 0.5

MISSING_CHANNEL = 0
NOISY_CHANNEL = 1
CONFLICT_CHANNEL = 2
STATE_CHANNEL = 6


@dataclass(frozen=True)
class CorrectionEvent:
    entity_id: int
    old_xy: tuple[float, float]
    new_xy: tuple[float, float]
    old_vel: tuple[float, float]
    new_vel: tuple[float, float]
    old_confidence: float
    new_confidence: float
    old_uncertainty: float
    new_uncertainty: float
    evidence_source: str
    correction_reason: str
    correction_magnitude: float
    evidence_weight: float
    belief_weight: float


def source_reliability(confidence: Tensor, age_seconds: Tensor, tau: float = RELIABILITY_TAU) -> Tensor:
    """Higher confidence and lower age → higher vote. Not a 50/50 average."""
    return confidence.clamp_min(1e-6) / (age_seconds.clamp_min(0.0) + tau)


def fuse(belief: Tensor, evidence: Tensor, w_belief: Tensor, w_evidence: Tensor) -> Tensor:
    denom = (w_belief + w_evidence).clamp_min(1e-8)
    return (w_belief * belief + w_evidence * evidence) / denom


def midpoint_is_wrong(belief: float, evidence: float, result: float) -> bool:
    mid = 0.5 * (belief + evidence)
    return abs(result - mid) > abs(result - evidence) * 0.25


@dataclass
class Revision:
    xy: Tensor
    vel: Tensor
    confidence: Tensor
    conflict: Tensor
    reason: str
    w_evidence: Tensor
    w_belief: Tensor
    event: CorrectionEvent | None


def revise_slot(
    *,
    entity_id: int,
    old_xy: Tensor,
    old_vel: Tensor,
    old_confidence: Tensor,
    old_uncertainty: Tensor,
    evidence_xy: Tensor,
    evidence_vel: Tensor,
    evidence_confidence: Tensor,
    evidence_uncertainty: Tensor,
    belief_age_seconds: Tensor,
    evidence_age_seconds: Tensor,
    observed_last_cycle: bool,
    evidence_source: str,
) -> Revision:
    """Update one entity belief.

    Consecutive observations (tracking): evidence gain ≈ confidence.
    Return after a gap (revision): reliability-weighted fuse, never 0.5/0.5.
    """
    delta_xy = torch.linalg.vector_norm(old_xy - evidence_xy)
    delta_vel = torch.linalg.vector_norm(old_vel - evidence_vel)
    delta = torch.maximum(delta_xy, delta_vel)
    if observed_last_cycle:
        gain = evidence_confidence.clamp(TRACKING_GAIN_FLOOR, 1.0)
        w_e = gain
        w_b = 1.0 - gain
        xy = (1.0 - gain) * old_xy + gain * evidence_xy
        vel = (1.0 - gain) * old_vel + gain * evidence_vel
        reason = "tracking"
    else:
        w_e = source_reliability(evidence_confidence, evidence_age_seconds)
        w_b = source_reliability(old_confidence, belief_age_seconds)
        xy = fuse(old_xy, evidence_xy, w_b, w_e)
        vel = fuse(old_vel, evidence_vel, w_b, w_e)
        reason = "evidence_dominance" if float(w_e) >= float(w_b) else "prior_retained"
        if float(delta) >= REVISION_MAGNITUDE:
            reason = "hypothesis_revision"

    conf = torch.maximum(old_confidence, evidence_confidence)
    if reason == "hypothesis_revision":
        conf = 0.5 * old_confidence + 0.5 * evidence_confidence
        conf = torch.minimum(conf + 0.2 * evidence_confidence, evidence_confidence.new_tensor(1.0))
    conflict = (delta_xy / 2.0).clamp(0.0, 1.0)
    if float(delta_vel) > float(delta_xy):
        conflict = (delta_vel / 2.0).clamp(0.0, 1.0)
    event = None
    if reason != "tracking" and float(delta) >= REVISION_MAGNITUDE:
        event = CorrectionEvent(
            entity_id=int(entity_id),
            old_xy=(float(old_xy[0]), float(old_xy[1])),
            new_xy=(float(xy[0]), float(xy[1])),
            old_vel=(float(old_vel[0]), float(old_vel[1])),
            new_vel=(float(vel[0]), float(vel[1])),
            old_confidence=float(old_confidence),
            new_confidence=float(conf),
            old_uncertainty=float(old_uncertainty.mean() if old_uncertainty.ndim > 0 else old_uncertainty),
            new_uncertainty=float(conflict),
            evidence_source=evidence_source,
            correction_reason=reason,
            correction_magnitude=float(delta),
            evidence_weight=float(w_e),
            belief_weight=float(w_b),
        )
    return Revision(
        xy=xy,
        vel=vel,
        confidence=conf,
        conflict=conflict,
        reason=reason,
        w_evidence=w_e if torch.is_tensor(w_e) else evidence_confidence.new_tensor(float(w_e)),
        w_belief=w_b if torch.is_tensor(w_b) else evidence_confidence.new_tensor(float(w_b)),
        event=event,
    )
