"""Belief revision: evidence updates current belief. Not memory. Not a future."""

from __future__ import annotations

import torch

from minakanushi.state.correction import fuse, revise_slot, source_reliability
from minakanushi.training.metrics import evidence_dominance


def test_stale_memory_loses_to_fresh_evidence() -> None:
    """Camera 130m conf 0.95 age 0.1s vs memory 100m conf 0.8 age 10s → ~127, not 115."""
    w_e = source_reliability(torch.tensor(0.95), torch.tensor(0.1))
    w_b = source_reliability(torch.tensor(0.80), torch.tensor(10.0))
    result = float(fuse(torch.tensor(100.0), torch.tensor(130.0), w_b, w_e))
    assert abs(result - 115.0) > 8.0
    assert evidence_dominance(result, 100.0, 130.0) == 1.0
    assert 124.0 < result < 130.0


def test_revision_after_gap_records_correction_event() -> None:
    old_xy = torch.tensor([2.5, 1.0])
    new_ev = torch.tensor([2.5, 1.0])
    old_vel = torch.tensor([-0.8, 0.0])
    ev_vel = torch.tensor([0.0, 0.0])
    out = revise_slot(
        entity_id=11,
        old_xy=old_xy,
        old_vel=old_vel,
        old_confidence=torch.tensor(0.9),
        old_uncertainty=torch.ones(8) * 0.2,
        evidence_xy=new_ev,
        evidence_vel=ev_vel,
        evidence_confidence=torch.tensor(0.95),
        evidence_uncertainty=torch.tensor(0.05),
        belief_age_seconds=torch.tensor(0.5),
        evidence_age_seconds=torch.tensor(0.0),
        observed_last_cycle=False,
        evidence_source="camera",
    )
    assert float(torch.linalg.vector_norm(out.vel)) < float(torch.linalg.vector_norm(old_vel))
    assert out.event is not None
    assert out.event.correction_reason in {"hypothesis_revision", "evidence_dominance"}
    assert out.event.old_vel[0] < -0.1
    assert abs(out.event.new_vel[0]) < abs(out.event.old_vel[0])
