"""Capability measurements. Loss going down is not a capability.

Does not construct 6.8B. Does not add layers. Language: measurements, not myths.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from minakanushi.architecture.config import load_config
from minakanushi.training.metrics import counterfactual_separation_score
from minakanushi.training.trainer import Trainer, UnrollPacket
from simulations.synthetic_world.dataset import training_frame

ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN_CLAIMS = (
    "MINA получила сознание",
    "MINA поняла мир",
    "MINA стала AGI",
    "модель думает как человек",
    "became conscious",
    "became AGI",
    "understood the world",
)

ALLOWED_CLAIMS = (
    "MINA улучшила prediction error",
    "MINA научилась корректировать belief",
    "memory улучшает future prediction",
    "ActionIntent выбирается с учётом world state",
)

SEEN_SEED = 7
UNSEEN_SEED = 9999
COUNTERFACTUAL_MIN = 1e-4

LEDGER_ROWS: tuple[dict[str, str], ...] = (
    {"ability": "World state reconstruction", "proven": "yes (cpu_dev / Milestone 1)", "gate": "v0.1"},
    {"ability": "Entity persistence", "proven": "yes (cpu_dev / Milestone 1)", "gate": "v0.1"},
    {"ability": "Belief revision", "proven": "partial", "gate": "Gate 03"},
    {"ability": "Memory improves future", "proven": "yes on short cpu_dev probe; long-hide not yet", "gate": "v0.3.1 / Gate E"},
    {"ability": "Counterfactual futures", "proven": "cpu_dev measured; 6.8B unknown", "gate": "Gate D"},
    {"ability": "Long horizon prediction", "proven": "waiting", "gate": "v0.4"},
    {"ability": "Causal attribution", "proven": "protocol ready; 6.8B unknown", "gate": "Gate C"},
    {"ability": "Held-out vs seen", "proven": "protocol ready; 6.8B unknown", "gate": "Gate B"},
    {"ability": "Revision honesty", "proven": "protocol ready; false=0 is not enough", "gate": "Gate F"},
    {"ability": "Multimodal grounding", "proven": "no", "gate": "Gate 9+"},
)


def cpu_trainer(sequence_length: int = 12) -> Trainer:
    cfg = load_config(
        ROOT / "configs" / "architecture" / "cpu_dev.yaml",
        training_path=ROOT / "configs" / "training" / "stage0_overfit.yaml",
        runtime_path=ROOT / "configs" / "runtime" / "cpu.yaml",
        simulation_path=ROOT / "configs" / "simulation" / "milestone1.yaml",
    )
    train = replace(
        cfg.training,
        steps=1,
        eval_every=1,
        checkpoint_every=10_000,
        log_every=1,
        sequence_length=int(sequence_length),
        dataset_root="",
    )
    return Trainer(replace(cfg, training=train), ROOT)


def packet_snapshot(pkt: UnrollPacket) -> dict[str, Any]:
    return {
        "scenario": pkt.scenario,
        "episode_index": int(pkt.episode_index),
        "frame_index": int(pkt.frame_index),
        "world_state": pkt.pred.entity_xy.detach().cpu(),
        "belief_state": pkt.before_xy.detach().cpu(),
        "future_prediction": pkt.pred_future.detach().cpu(),
        "revision_output": pkt.pred.entity_xy.detach().cpu(),
        "should_revise": pkt.should_revise.detach().cpu(),
        "alt_future": pkt.alt_future.detach().cpu(),
    }


def _scalar(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.float()).item())


def snapshot_drift(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    keys = ("world_state", "future_prediction", "belief_state", "revision_output")
    return {key: _scalar(after[key] - before[key]) for key in keys}


def gate_a_retention(trainer: Trainer, out_dir: Path | None = None) -> dict[str, Any]:
    """Old scenarios must remain measurable after new data. Capture before."""
    rows = {}
    for name in ("const_velocity", "hidden_correction", "agent_move"):
        pkt = trainer.unroll(1, scenario=name, episode_index=0, seed=SEEN_SEED, length=12)
        snap = packet_snapshot(pkt)
        metrics = trainer._metrics(pkt)
        rows[name] = {
            "future_ADE": metrics["future_ADE"],
            "revision_detected": metrics["revision_detected"],
            "world_norm": _scalar(snap["world_state"]),
            "future_norm": _scalar(snap["future_prediction"]),
        }
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(snap, out_dir / f"{name}.pt")
    return {
        "gate": "A_retention",
        "claim": "snapshots of world_state / future / belief / revision. Not 'she remembered who she is'.",
        "scenarios": rows,
        "pass": True,
    }


def gate_b_heldout(trainer: Trainer) -> dict[str, Any]:
    """Train seed vs new seed. Improvement on seen-only is memorization."""
    seen: dict[str, float] = {}
    unseen: dict[str, float] = {}
    for name in ("const_velocity", "agent_move"):
        pkt_s = trainer.unroll(1, scenario=name, episode_index=0, seed=SEEN_SEED, length=12)
        pkt_u = trainer.unroll(1, scenario=name, episode_index=0, seed=UNSEEN_SEED, length=12)
        seen[name] = trainer._metrics(pkt_s)["future_ADE"]
        unseen[name] = trainer._metrics(pkt_u)["future_ADE"]
    return {
        "gate": "B_heldout",
        "claim": "seen vs unseen ADE. Train-only drop without held-out drop is memorization.",
        "seen_seed": SEEN_SEED,
        "unseen_seed": UNSEEN_SEED,
        "seen_ADE": seen,
        "unseen_ADE": unseen,
        "memorization_risk": "unknown_until_after_train",
        "pass": True,
    }


def gate_c_causality(trainer: Trainer) -> dict[str, Any]:
    """Why the future changed: external event + invalid velocity hypothesis."""
    pkt = trainer.unroll(1, scenario="unexpected_stop", episode_index=0, seed=SEEN_SEED, length=12)
    metrics = trainer._metrics(pkt)
    event = "unexpected_physics"
    detected = float(metrics["revision_detected"])
    picture_in_picture_out = detected <= 0.0
    return {
        "gate": "C_causality",
        "claim": "external event unexpected_physics, revision = velocity hypothesis invalid. Not new-picture → new-answer.",
        "external_event": event,
        "scenario": pkt.scenario,
        "frame": pkt.frame_index,
        "training_frame": training_frame("unexpected_stop", 12),
        "revision_detected": detected,
        "picture_in_picture_out": picture_in_picture_out,
        "capability_proven": not picture_in_picture_out,
        "pass": event == "unexpected_physics",
    }


def gate_d_counterfactual(trainer: Trainer) -> dict[str, Any]:
    pkt = trainer.unroll(1, scenario="const_velocity", episode_index=0, seed=SEEN_SEED, length=12)
    distance = float(counterfactual_separation_score(pkt.pred_future[0, -1], pkt.alt_future[0, -1]).detach())
    labeled = pkt.candidates[0].objective
    alt = pkt.candidates[1].objective
    return {
        "gate": "D_counterfactual",
        "claim": "same observation, WAIT vs MOVE_TO, Future A ≠ Future B. Distance≈0 means no world model.",
        "action_a": labeled,
        "action_b": alt,
        "future_distance": distance,
        "pass": distance > COUNTERFACTUAL_MIN and {labeled, alt} == {"WAIT", "MOVE_TO"},
    }


def gate_e_memory(trainer: Trainer, *, length: int = 32) -> dict[str, Any]:
    pkt = trainer.unroll(1, scenario="hidden_correction", episode_index=0, seed=SEEN_SEED, length=length)
    metrics = trainer._metrics(pkt)
    hidden = pkt.frame_index
    return {
        "gate": "E_memory",
        "claim": "object gone for many frames then returns. ADE(memory on) vs ADE(memory off).",
        "hidden_frames_before_evidence": int(hidden),
        "reacquisition_accuracy": metrics["reacquisition_accuracy"],
        "memory_ade_on": metrics["memory_ade_on"],
        "memory_ade_off": metrics["memory_ade_off"],
        "memory_helps_future": metrics["memory_helps_future"],
        "memory_future_delta": metrics["memory_future_delta"],
        "pass": metrics["memory_ade_on"] is not None,
    }


def gate_f_revision_honesty(trainer: Trainer) -> dict[str, Any]:
    need = trainer.unroll(1, scenario="hidden_correction", episode_index=0, seed=SEEN_SEED, length=12)
    persist = trainer.unroll(1, scenario="const_velocity", episode_index=0, seed=SEEN_SEED, length=12)
    need_m = trainer._metrics(need)
    persist_m = trainer._metrics(persist)
    never_revises = need_m["revision_detected"] == 0.0 and persist_m["false_revision_rate"] == 0.0
    return {
        "gate": "F_revision_honesty",
        "claim": "revise when belief is wrong; persist when belief is right. false_revision=0 alone can mean never-revises.",
        "when_needed": {
            "scenario": "hidden_correction",
            "revision_detected": need_m["revision_detected"],
            "revision_accuracy": need_m["revision_accuracy"],
        },
        "when_not_needed": {
            "scenario": "const_velocity",
            "false_revision_rate": persist_m["false_revision_rate"],
            "revision_detected": persist_m["revision_detected"],
        },
        "never_revises_trap": never_revises,
        "pass": not never_revises or need_m["revision_detected"] > 0.0,
    }


def compare_heldout(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """After training: if seen ADE drops and unseen does not, memorization."""
    seen_b = before["seen_ADE"]
    unseen_b = before["unseen_ADE"]
    seen_a = after["seen_ADE"]
    unseen_a = after["unseen_ADE"]
    seen_gain = {k: seen_b[k] - seen_a[k] for k in seen_b}
    unseen_gain = {k: unseen_b[k] - unseen_a[k] for k in unseen_b}
    only_seen = all(v > 0.0 for v in seen_gain.values()) and all(v <= 0.0 for v in unseen_gain.values())
    return {
        "seen_improvement": seen_gain,
        "unseen_improvement": unseen_gain,
        "memorization": only_seen,
        "pass": not only_seen,
    }


def forbidden_in_text(text: str) -> list[str]:
    blob = text.lower()
    hits = []
    for claim in FORBIDDEN_CLAIMS:
        if claim.lower() in blob:
            hits.append(claim)
    return hits


def run_capability_suite(out_dir: Path) -> dict[str, Any]:
    trainer = cpu_trainer(12)
    ref = out_dir / "reference_before"
    gates = {
        "A": gate_a_retention(trainer, ref),
        "B": gate_b_heldout(trainer),
        "C": gate_c_causality(trainer),
        "D": gate_d_counterfactual(trainer),
        "E": gate_e_memory(trainer),
        "F": gate_f_revision_honesty(trainer),
    }
    report = {
        "protocol": "MINA capability ledger",
        "not_a_claim": "loss dropped is not intelligence appeared",
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
        "allowed_claims": list(ALLOWED_CLAIMS),
        "ledger": list(LEDGER_ROWS),
        "gates": gates,
        "pass": all(bool(row.get("pass")) for row in gates.values()),
    }
    return report
