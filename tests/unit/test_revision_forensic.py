"""v0.3.1-R: sensor_delay revision cut is teacher / frame calibration, not a new train."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from helpers import cpu_config
from minakanushi.architecture.mina_unit import KIND_IDS
from minakanushi.state.correction import REVISION_MAGNITUDE
from minakanushi.training.revision_forensic import (
    classify_cut,
    generated_sensor_delay_geometry,
    live_slot_audit,
    metric_collapses_when_teacher_empty,
    spatial_disagreement,
    thresholds,
)
from simulations.synthetic_world.dataset import generate_episode, training_frame


def test_thresholds_are_the_frozen_cut() -> None:
    t = thresholds()
    assert t["revision_magnitude"] == 0.25
    assert t["move_detect"] == 0.05
    assert t["sensor_delay_s"] == 0.15


def test_empty_teacher_is_scored_as_missed_detection() -> None:
    ident = metric_collapses_when_teacher_empty()
    assert ident["n_need"] == 0
    assert ident["revision_detected"] == 0.0
    assert ident["collapses_to_zero"] is True


def test_sensor_delay_train_frame_loses_the_mover() -> None:
    sim = cpu_config().simulation
    episode = generate_episode(sim, seed=11, episode_index=3, length=32, scenario="sensor_delay")
    geo = spatial_disagreement(episode)
    train_idx = training_frame(episode.scenario, len(episode.observations))
    obs = episode.observations[train_idx]
    assert obs.arrival_time is not None
    assert abs(float(obs.arrival_time) - float(obs.timestamp) - 0.15) < 1e-9
    assert geo["n_visible_movers"] == 0
    assert geo["train_frame_after_mover_left"] is True
    assert geo["early"]["n_visible_movers"] >= 1
    assert geo["early"]["visible_xy_is_current"] is True
    assert geo["early"]["teacher_if_belief_is_truth_prev"] is False
    assert geo["early"]["delay_path"] < REVISION_MAGNITUDE
    assert geo["cut"] == "train_frame_has_no_mover_evidence"


def test_generated_bundle_cuts_at_train_frame_not_at_delay_stamp() -> None:
    bundle = generated_sensor_delay_geometry()
    assert bundle["train_frame_no_mover_rate"] == 1.0
    assert bundle["early_mover_visible_rate"] == 1.0
    assert bundle["obs_is_current_xy_rate"] == 1.0
    assert bundle["teacher_if_oracle_prev_rate"] == 0.0
    assert bundle["early_teacher_if_oracle_prev_rate"] == 0.0
    assert bundle["mean_step"] < REVISION_MAGNITUDE
    assert bundle["mean_delay_path"] < REVISION_MAGNITUDE


def test_better_prediction_classifies_as_teacher_suppressed() -> None:
    assert classify_cut(n_need=0, max_before_d=0.08, detected=0.0, n_mover_evidence=0) == "no_mover_evidence"
    assert classify_cut(n_need=0, max_before_d=0.08, detected=0.0, n_mover_evidence=1) == "teacher_suppressed"
    assert classify_cut(n_need=1, max_before_d=0.40, detected=0.0, n_mover_evidence=1) == "model_did_not_move"
    assert classify_cut(n_need=1, max_before_d=0.40, detected=1.0, n_mover_evidence=1) == "trigger_live"


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
        frame_index=16,
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
    assert audit["better_prediction_suppresses_teacher"] is True
