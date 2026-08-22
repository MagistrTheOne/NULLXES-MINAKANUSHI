"""v0.3.1-R patch: frame, metric, delay teacher. No 6.8B. No train."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from helpers import cpu_config
from minakanushi.architecture.mina_unit import KIND_IDS
from minakanushi.state.correction import REVISION_MAGNITUDE
from minakanushi.training.revision import revision_metrics, should_revise_mask
from minakanushi.training.revision_forensic import (
    classify_cut,
    diagnose,
    generated_sensor_delay_geometry,
    live_slot_audit,
    metric_empty_teacher_is_not_a_miss,
    spatial_disagreement,
)
from simulations.synthetic_world.dataset import generate_episode, training_frame


def _slots(*, before, evidence, after, should, has=None, occ=None, ids=None):
    has = has if has is not None else torch.tensor([[True, True]])
    occ = occ if occ is not None else torch.tensor([[True, True]])
    ids = ids if ids is not None else torch.tensor([[11, 1]])
    return revision_metrics(
        before_xy=before,
        after_xy=after,
        evidence_xy=evidence,
        should_revise=should,
        has_evidence=has,
        occupied_before=occ,
        entity_id=ids,
    )


def test_empty_teacher_is_excluded_from_detection_denominator() -> None:
    ident = metric_empty_teacher_is_not_a_miss()
    assert ident["n_need"] == 0
    assert ident["excluded_from_detection"] is True
    assert math.isnan(ident["revision_detected"])


def test_empty_teacher_no_move_is_not_false_revision() -> None:
    before = torch.zeros(1, 2, 2)
    evidence = torch.zeros(1, 2, 2)
    after = before.clone()
    should = torch.tensor([[False, False]])
    metrics = _slots(before=before, evidence=evidence, after=after, should=should)
    assert metrics["n_need"] == 0.0
    assert math.isnan(metrics["revision_detected"])
    assert metrics["false_revision_rate"] == 0.0
    assert metrics["n_false_revision"] == 0.0


def test_empty_teacher_with_move_is_false_revision() -> None:
    before = torch.zeros(1, 2, 2)
    evidence = torch.zeros(1, 2, 2)
    after = torch.tensor([[[0.40, 0.0], [0.0, 0.0]]])
    should = torch.tensor([[False, False]])
    metrics = _slots(before=before, evidence=evidence, after=after, should=should)
    assert metrics["n_need"] == 0.0
    assert math.isnan(metrics["revision_detected"])
    assert metrics["false_revision_rate"] == 1.0
    assert metrics["n_false_revision"] == 1.0


def test_sensor_delay_uses_mover_visible_frame_not_mid() -> None:
    sim = cpu_config().simulation
    episode = generate_episode(sim, seed=11, episode_index=3, length=64, scenario="sensor_delay")
    idx = training_frame(episode.scenario, len(episode.observations))
    geo = spatial_disagreement(episode)
    assert idx == 2
    assert idx != len(episode.observations) // 2
    assert idx != 32
    assert geo["n_visible_movers"] >= 1
    assert geo["training_frame"] == 2


def test_static_obstacle_residual_cannot_create_sensor_delay_teacher() -> None:
    before = torch.tensor([[[3.0, 0.0], [0.0, 0.0]]])
    evidence = torch.zeros(1, 2, 2)
    has = torch.tensor([[True, True]])
    occ = torch.tensor([[True, True]])
    ids = torch.tensor([[21, 1]])
    obstacle = torch.tensor([[KIND_IDS["obstacle"], KIND_IDS["agent"]]])
    mover = torch.tensor([[KIND_IDS["mover"], KIND_IDS["agent"]]])
    unconstrained = should_revise_mask(before, evidence, has, occ, ids)
    assert bool(unconstrained[0, 0])
    delayed_obs = should_revise_mask(
        before, evidence, has, occ, ids, entity_kind=obstacle, scenario="sensor_delay"
    )
    assert not bool(delayed_obs.any())
    delayed_mover = should_revise_mask(
        before, evidence, has, occ, ids, entity_kind=mover, scenario="sensor_delay"
    )
    assert bool(delayed_mover[0, 0])


def test_hidden_correction_and_conflict_scoring_unchanged() -> None:
    before = torch.tensor([[[4.53, 2.25], [1.0, 1.0]]])
    evidence = torch.tensor([[[2.44, 0.92], [1.0, 1.0]]])
    after = evidence.clone()
    zeros = torch.zeros_like(before)
    ids = torch.tensor([[11, 1]])
    occ = torch.tensor([[True, True]])
    has = torch.tensor([[True, True]])
    should = should_revise_mask(before, evidence, has, occ, ids, scenario="hidden_correction")
    assert bool(should[0, 0])
    assert not bool(should[0, 1])
    metrics = revision_metrics(
        before_xy=before,
        after_xy=after,
        evidence_xy=evidence,
        should_revise=should,
        has_evidence=has,
        occupied_before=occ,
        entity_id=ids,
    )
    assert metrics["n_need"] == 1.0
    assert metrics["revision_detected"] == 1.0
    assert metrics["revision_required_recall"] == 1.0
    assert metrics["revision_direction_accuracy"] == 1.0
    assert training_frame("hidden_correction", 12) == 6
    assert training_frame("hidden_correction", 64) == 34
    assert training_frame("conflict", 12) == 4
    assert training_frame("conflict", 32) > 4
    assert training_frame("const_velocity", 12) == 6
    _ = zeros


def test_generated_bundle_uses_frame_two() -> None:
    bundle = generated_sensor_delay_geometry(length=64)
    assert bundle["train_frame_no_mover_rate"] == 0.0
    assert all(r["training_frame"] == 2 for r in bundle["rows"])
    assert all(r["n_visible_movers"] >= 1 for r in bundle["rows"])


def test_cpu_forensic_patch_pass() -> None:
    report = diagnose()
    cpu = report["cpu_verdict"]
    assert cpu["sensor_delay_frame"] == 2
    assert cpu["sensor_delay_frame_not_mid"] is True
    assert cpu["mover_evidence_at_train_frame"] is True
    assert cpu["empty_teacher_not_missed_detection"] is True
    assert cpu["cpu_patch_pass"] is True
    assert cpu["accepted"] is False
    assert cpu["variant"] == "B"


def test_live_slot_audit_flags_suppressed_teacher() -> None:
    before = torch.zeros(1, 2, 2)
    evidence = torch.tensor([[[0.06, 0.0], [0.0, 0.0]]])
    after = before.clone()
    ids = torch.tensor([[11, 1]])
    has = torch.tensor([[True, True]])
    occ = torch.tensor([[True, True]])
    should = torch.tensor([[False, False]])
    pkt = SimpleNamespace(
        scenario="sensor_delay",
        frame_index=2,
        before_xy=before,
        evidence_xy=evidence,
        has_evidence=has,
        occupied_before=occ,
        should_revise=should,
        n_constructor_corrections=0,
        pred=SimpleNamespace(
            entity_xy=after,
            entity_id=ids,
            kind=torch.tensor([[KIND_IDS["mover"], KIND_IDS["agent"]]]),
            xy_std=torch.ones(1, 2, 2) * 0.2,
            uncertainty=torch.zeros(1, 2, 8),
        ),
    )
    audit = live_slot_audit(pkt)
    assert audit["n_need"] == 0
    assert audit["n_mover_evidence"] == 1
    assert audit["cut"] == "teacher_suppressed"
    assert math.isnan(audit["revision_detected"])
    assert classify_cut(n_need=0, max_before_d=0.08, detected=float("nan"), n_mover_evidence=1) == "teacher_suppressed"
    assert REVISION_MAGNITUDE == 0.25
