"""Belief-revision training signal — curriculum, frame, L_revision, metrics."""

from __future__ import annotations

import torch

from helpers import ROOT, cpu_config
from minakanushi.architecture.config import LossLambdaConfig, TrainingConfig, load_training
from minakanushi.training.objectives import compute_objectives
from minakanushi.training.revision import revision_losses, should_revise_mask
from minakanushi.training.trainer import trainer_from_files
from simulations.synthetic_world.dataset import (
    TRAIN_CURRICULUM,
    generate_episode,
    generate_overfit_set,
    training_frame,
)
from simulations.synthetic_world.dataset_v1 import SPLIT_SCENARIOS


def test_curriculum_contains_gate03_as_training_not_ood_only() -> None:
    required = ("hidden_correction", "conflict", "reacquisition", "gone_forever")
    for name in required:
        assert name in TRAIN_CURRICULUM
        assert name in SPLIT_SCENARIOS["train"]
    assert "conflict" in SPLIT_SCENARIOS["ood"]


def test_training_frame_is_the_correction_event() -> None:
    assert training_frame("hidden_correction", 12) == 6
    assert training_frame("reacquisition", 12) == 6
    assert training_frame("conflict", 12) == 4
    assert training_frame("gone_forever", 12) == 3
    assert training_frame("hidden_correction", 6) == 4
    assert training_frame("const_velocity", 12) == 6
    assert training_frame("hidden_correction", 64) == 34
    assert training_frame("conflict", 32) > 4


def test_reacquisition_is_hidden_correction_physics() -> None:
    sim = cpu_config().simulation
    ep = generate_episode(sim, seed=7, episode_index=0, length=12, scenario="reacquisition")
    assert ep.scenario == "reacquisition"
    mover = None
    for i, eid in enumerate(ep.truth[0].entity_id):
        if ep.truth[0].kind[i] == "mover":
            mover = int(eid)
            break
    assert mover is not None
    assert mover not in ep.truth[3].visible_ids
    assert mover in ep.truth[6].visible_ids


def test_overfit_set_cycles_curriculum() -> None:
    sim = cpu_config().simulation
    eps = generate_overfit_set(sim, seed=7, n_episodes=8, length=12)
    names = [e.scenario for e in eps]
    assert names == list(TRAIN_CURRICULUM)


def test_revision_losses_penalize_wrong_direction() -> None:
    before = torch.tensor([[[4.53, 2.25], [1.0, 1.0]]])
    evidence = torch.tensor([[[2.44, 0.92], [1.0, 1.0]]])
    after_wrong = before.clone()
    after_away = before + (before - evidence)
    after_right = evidence.clone()
    zeros = torch.zeros_like(before)
    ids = torch.tensor([[11, 1]])
    occ = torch.tensor([[True, True]])
    has = torch.tensor([[True, True]])
    should = should_revise_mask(before, evidence, has, occ, ids)
    assert bool(should[0, 0])
    assert not bool(should[0, 1])
    wrong, parts_w = revision_losses(
        before_xy=before,
        after_xy=after_wrong,
        after_vel=zeros,
        evidence_xy=evidence,
        evidence_vel=zeros,
        should_revise=should,
        has_evidence=has,
        occupied_before=occ,
        entity_id=ids,
    )
    away, parts_a = revision_losses(
        before_xy=before,
        after_xy=after_away,
        after_vel=zeros,
        evidence_xy=evidence,
        evidence_vel=zeros,
        should_revise=should,
        has_evidence=has,
        occupied_before=occ,
        entity_id=ids,
    )
    right, parts_r = revision_losses(
        before_xy=before,
        after_xy=after_right,
        after_vel=zeros,
        evidence_xy=evidence,
        evidence_vel=zeros,
        should_revise=should,
        has_evidence=has,
        occupied_before=occ,
        entity_id=ids,
    )
    assert float(wrong) > float(right)
    assert float(away) > float(right)
    assert float(parts_w["revision_detection"]) > float(parts_r["revision_detection"])
    assert float(parts_w["revision_calibration"]) > float(parts_r["revision_calibration"])
    assert float(parts_a["revision_direction"]) > float(parts_r["revision_direction"])
    assert float(parts_a["revision_direction"]) > 0.0


def test_revision_term_absent_when_tensors_omitted() -> None:
    train = TrainingConfig(lambdas=LossLambdaConfig(revision=1.0))
    dummy = torch.zeros(1, 2, 2)
    occ = torch.ones(1, 2, dtype=torch.bool)
    future = dummy.unsqueeze(1).expand(1, 2, 2, 2).contiguous()
    unc = torch.ones(1, 2, 8) * 0.2
    breakdown = compute_objectives(
        pred_xy=dummy,
        true_xy=dummy,
        occupied=occ,
        pred_next_xy=dummy,
        true_next_xy=dummy,
        pred_future_xy=future,
        true_future_xy=future,
        uncertainty=unc,
        memory_xy=dummy,
        memory_true_xy=dummy,
        memory_mask=occ,
        causal_pred=dummy,
        causal_true=dummy,
        alt_future_xy=future,
        intra_branch_xy=future,
        latent=torch.zeros(1, 2, 8),
        training=train,
    )
    assert "revision" in breakdown.terms
    assert float(breakdown.terms["revision"]) == 0.0


def test_stage0_yaml_wires_revision_lambda() -> None:
    cfg = load_training(ROOT / "configs" / "training" / "stage0_overfit.yaml")
    assert cfg.lambdas.revision == 1.0


def test_trainer_unroll_hits_correction_event() -> None:
    trainer = trainer_from_files(ROOT, ROOT / "configs" / "training" / "stage0_overfit.yaml")
    pkt = trainer.unroll(1)
    assert pkt.scenario == "hidden_correction"
    assert pkt.frame_index == 6
    assert pkt.n_constructor_corrections >= 1
    assert bool(pkt.should_revise.any())
    assert "revision" in pkt.breakdown.terms
    assert "revision_detection" in pkt.breakdown.terms
    metrics = trainer._metrics(pkt)
    assert "revision_detected" in metrics
    assert "revision_direction_accuracy" in metrics
    assert "revision_magnitude_error" in metrics
    assert "revision_latency" in metrics
    assert "false_revision_rate" in metrics
    assert metrics["belief_revision_accuracy"] == metrics["revision_direction_accuracy"]
