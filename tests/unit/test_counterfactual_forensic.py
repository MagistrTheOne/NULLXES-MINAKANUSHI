"""Official cf dilutes a real agent delta across empty world slots. No 6.8B."""

from __future__ import annotations

import torch

from minakanushi.training.counterfactual_forensic import (
    V031_FUTURE_FROBENIUS,
    V031_OFFICIAL_CF_MEAN,
    embodiment_audit,
    metric_collapse_report,
    replay_sampler,
)
from minakanushi.training.metrics import counterfactual_layers, counterfactual_separation_score


def test_official_cf_is_agent_delta_over_empty_slots() -> None:
    slots = 512
    agent = 0.40
    a = torch.zeros(slots, 2)
    b = torch.zeros(slots, 2)
    b[0, 0] = agent
    occ = torch.zeros(slots)
    occ[0] = 1.0
    official = float(counterfactual_separation_score(a, b))
    layers = counterfactual_layers(a, b, occ)
    assert abs(official - agent / slots) < 1e-6
    assert abs(float(layers["agent_cf"]) - agent) < 1e-6
    assert abs(float(layers["occupied_cf"]) - agent) < 1e-6
    assert abs(float(layers["frobenius"]) - agent) < 1e-6
    assert abs(official * slots - agent) < 1e-6
    assert abs(float(layers["empty_slot_dilution"]) - 1.0 / slots) < 1e-6
    assert abs(float(layers["empty_slot_factor"]) - slots) < 1e-6
    assert abs(float(layers["empty_slot_collapse"]) - slots) < 1e-6
    assert abs(float(layers["empty_slot_collapse"]) - 1.0) > 0.5


def test_v031_ledger_recovers_future_engine_norm() -> None:
    report = metric_collapse_report()
    assert report["matches_identity"] is True
    assert report["fork"] == "A"
    recovered = V031_OFFICIAL_CF_MEAN * 512
    assert abs(recovered - V031_FUTURE_FROBENIUS) < 0.02
    assert report["v031_ledger"]["recovered_diversity_would_pass"] is True


def test_sampler_replay_is_deterministic() -> None:
    phases = tuple(["physics", "agency", "causality", "embodiment"] * 4)
    scenarios = (
        "accelerate",
        "agent_move",
        "gone_forever",
        "sensor_delay",
    ) * 4
    a = replay_sampler(phases, scenarios, first_step=1, last_step=32, resume_start=1, warm_steps=4)
    b = replay_sampler(phases, scenarios, first_step=1, last_step=32, resume_start=1, warm_steps=4)
    assert a["scenarios"] == b["scenarios"]
    assert a["paired_wait_move_every_step"] == 32
    assert a["first_step"] == 1
    assert a["optimizer_steps"][0]["step"] == 1
    assert sum(a["phases"].values()) == 32


def test_v031_replay_uses_global_steps_129_to_1128_not_1_to_1000() -> None:
    phases = tuple(["physics", "agency", "causality", "embodiment"] * 8)
    scenarios = ("accelerate", "agent_move", "gone_forever", "sensor_delay") * 8
    wrong = replay_sampler(phases, scenarios, first_step=1, last_step=1000, resume_start=1)
    right = replay_sampler(phases, scenarios)
    assert right["first_step"] == 129
    assert right["last_step"] == 1128
    assert right["optimizer_steps"][0]["step"] == 129
    assert right["optimizer_steps"][0]["job"] == 1
    assert right["optimizer_steps"][-1]["step"] == 1128
    assert right["optimizer_steps"][-1]["job"] == 1000
    assert wrong["scenarios"] != right["scenarios"]


def test_embodiment_audit_does_not_blame_correction_slices() -> None:
    rows = [
        {"phase": "causality", "scenario": "hidden_correction", "revision_detected": 1.0, "revision_direction_accuracy": 1.0, "false_revision_rate": 0.0},
        {"phase": "embodiment", "scenario": "sensor_delay", "revision_detected": 0.0, "revision_direction_accuracy": 0.0, "false_revision_rate": 0.0},
        {"phase": "causality", "scenario": "gone_forever", "revision_detected": 0.0, "revision_direction_accuracy": 0.0, "false_revision_rate": 1.0},
        {"phase": "causality", "scenario": "gone_forever", "revision_detected": 0.0, "revision_direction_accuracy": 0.0, "false_revision_rate": 1.0},
        {"phase": "causality", "scenario": "gone_forever", "revision_detected": 1.0, "revision_direction_accuracy": 1.0, "false_revision_rate": 0.0},
    ]
    audit = embodiment_audit(rows)
    assert audit["slices"]["hidden_correction"]["detection"] == 1.0
    assert audit["slices"]["sensor_delay"]["detection"] == 0.0
    assert audit["gone_forever_n_is_small"] is True
    assert audit["do_not_change_global_objective"] is True
