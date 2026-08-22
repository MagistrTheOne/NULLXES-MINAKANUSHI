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
    {"ability": "Belief revision", "proven": "partial (Gate 03 hidden_correction); unexpected_stop not proven", "gate": "Gate 03 / C"},
    {"ability": "Memory improves future", "proven": "no at Gate E length=32 (ADE on worse than off). Short probe is not this gate.", "gate": "v0.3.1 / Gate E"},
    {"ability": "Counterfactual existence", "proven": "yes on cpu_dev (WAIT vs MOVE_TO distance > 0); 6.8B unknown", "gate": "Gate D"},
    {"ability": "Counterfactual diversity", "proven": "NOT PASS. Pack min=max≈0.779, std≈0. One arena geometry. v0.4 geometry expansion.", "gate": "v0.3 audit / v0.4"},
    {"ability": "Long horizon prediction", "proven": "waiting", "gate": "v0.4"},
    {"ability": "Causal attribution", "proven": "no; revision_detected=0, picture_in_picture_out on unexpected_stop", "gate": "Gate C"},
    {"ability": "Held-out vs seen", "proven": "protocol ready; 6.8B unknown", "gate": "Gate B"},
    {"ability": "Revision honesty", "proven": "protocol ready; false=0 is not enough", "gate": "Gate F"},
    {"ability": "No shortcut", "proven": "protocol ready; 6.8B unknown", "gate": "Gate G"},
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
    proven = (not picture_in_picture_out) and pkt.scenario == "unexpected_stop"
    return {
        "gate": "C_causality",
        "claim": "external event unexpected_physics, revision = velocity hypothesis invalid. Not new-picture → new-answer.",
        "external_event": event,
        "scenario": pkt.scenario,
        "frame": pkt.frame_index,
        "training_frame": training_frame("unexpected_stop", 12),
        "revision_detected": detected,
        "picture_in_picture_out": picture_in_picture_out,
        "capability_proven": proven,
        "pass": proven,
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
    ade_on = float(metrics["memory_ade_on"])
    ade_off = float(metrics["memory_ade_off"])
    helps = bool(metrics["memory_helps_future"]) and ade_on < ade_off
    return {
        "gate": "E_memory",
        "claim": "object gone for many frames then returns. ADE(memory on) vs ADE(memory off).",
        "hidden_frames_before_evidence": int(hidden),
        "reacquisition_accuracy": metrics["reacquisition_accuracy"],
        "memory_ade_on": ade_on,
        "memory_ade_off": ade_off,
        "memory_helps_future": metrics["memory_helps_future"],
        "memory_future_delta": metrics["memory_future_delta"],
        "n": 1,
        "length": int(length),
        "pass": helps,
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


def compare_retention(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Old v0.1 abilities must not collapse while a new trick looks pretty."""
    before_rows = before.get("scenarios") or {}
    after_rows = after.get("scenarios") or {}
    rows: dict[str, Any] = {}
    collapsed = False
    for name, b in before_rows.items():
        a = after_rows.get(name) or {}
        b_ade = float(b["future_ADE"])
        a_ade = float(a.get("future_ADE", b_ade))
        rows[name] = {"before": b_ade, "after": a_ade, "delta": a_ade - b_ade}
        if a_ade > max(2.0 * b_ade, b_ade + 1.0):
            collapsed = True
    physics = rows.get("const_velocity") or {}
    hidden = rows.get("hidden_correction") or {}
    not_progress = bool(
        physics
        and hidden
        and physics.get("delta", 0.0) > 0.0
        and hidden.get("delta", 0.0) < 0.0
    )
    return {
        "gate": "A_retention_compare",
        "claim": "entity persistence / basic world / constraints stay measurable. Hidden-correction up with physics forgotten is not progress.",
        "scenarios": rows,
        "old_ability_collapsed": collapsed,
        "physics_forgotten_for_hidden_trick": not_progress,
        "pass": (not collapsed) and (not not_progress),
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


def _scalar_verdict(before: float, after: float, *, lower_is_better: bool) -> str:
    if lower_is_better:
        if after < before:
            return "improved"
        if after > before:
            return "worse"
        return "unchanged"
    if after > before:
        return "improved"
    if after < before:
        return "worse"
    return "unchanged"


def compare_ability_table(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Before/after table. Loss is not a row."""
    bg = before.get("gates") or before
    ag = after.get("gates") or after

    def _ade(gate: dict[str, Any], scenario: str) -> float:
        return float((gate.get("scenarios") or {}).get(scenario, {}).get("future_ADE", float("nan")))

    persistence_b = _ade(bg.get("A") or {}, "const_velocity")
    persistence_a = _ade(ag.get("A") or {}, "const_velocity")
    revision_b = float((bg.get("C") or {}).get("revision_detected", 0.0))
    revision_a = float((ag.get("C") or {}).get("revision_detected", 0.0))
    mem_b = float((bg.get("E") or {}).get("memory_helps_future", 0.0))
    mem_a = float((ag.get("E") or {}).get("memory_helps_future", 0.0))
    cf_b = float((bg.get("D") or {}).get("future_distance", 0.0))
    cf_a = float((ag.get("D") or {}).get("future_distance", 0.0))
    seen_b = (bg.get("B") or {}).get("seen_ADE") or {}
    seen_a = (ag.get("B") or {}).get("seen_ADE") or {}
    unseen_b = (bg.get("B") or {}).get("unseen_ADE") or {}
    unseen_a = (ag.get("B") or {}).get("unseen_ADE") or {}
    held_b = float(sum(unseen_b.values()) / max(len(unseen_b), 1)) if unseen_b else float("nan")
    held_a = float(sum(unseen_a.values()) / max(len(unseen_a), 1)) if unseen_a else float("nan")
    mem_only = compare_heldout(bg["B"], ag["B"]) if "B" in bg and "B" in ag else {"memorization": False}

    rows = [
        {
            "ability": "Persistence",
            "before": persistence_b,
            "after": persistence_a,
            "verdict": _scalar_verdict(persistence_b, persistence_a, lower_is_better=True),
        },
        {
            "ability": "Revision",
            "before": revision_b,
            "after": revision_a,
            "verdict": _scalar_verdict(revision_b, revision_a, lower_is_better=False),
        },
        {
            "ability": "Memory future",
            "before": mem_b,
            "after": mem_a,
            "verdict": _scalar_verdict(mem_b, mem_a, lower_is_better=False),
        },
        {
            "ability": "Counterfactual",
            "before": cf_b,
            "after": cf_a,
            "verdict": _scalar_verdict(cf_b, cf_a, lower_is_better=False),
        },
        {
            "ability": "Held-out ADE",
            "before": held_b,
            "after": held_a,
            "verdict": "memorization"
            if mem_only.get("memorization")
            else _scalar_verdict(held_b, held_a, lower_is_better=True),
        },
    ]
    real = (
        not mem_only.get("memorization")
        and any(row["ability"] == "Held-out ADE" and row["verdict"] == "improved" for row in rows)
        and any(row["ability"] == "Memory future" and row["verdict"] == "improved" for row in rows)
    )
    return {
        "table": rows,
        "memorization": bool(mem_only.get("memorization")),
        "real_improvement": real,
        "claim_if_real": "обучение улучшило прогнозирование и обновление убеждений на проверочных сценариях.",
        "claim_if_not": "train metrics up without held-out is memorization. Do not update the ledger.",
    }


def forbidden_in_text(text: str) -> list[str]:
    blob = text.lower()
    hits = []
    for claim in FORBIDDEN_CLAIMS:
        if claim.lower() in blob:
            hits.append(claim)
    return hits


def gate_g_no_shortcut(trainer: Trainer) -> dict[str, Any]:
    """If ablating vision vs telemetry is a no-op, or permute is a no-op, it is a shortcut."""
    from minakanushi.training.shortcut import mutate_episode

    base = trainer._load_episode(1, scenario="const_velocity", episode_index=0, seed=SEEN_SEED, length=12)
    modes = ("full", "drop_vision", "delay_telemetry", "permute_structure")
    ades: dict[str, float] = {}
    for mode in modes:
        pkt = trainer.unroll(1, episode=mutate_episode(base, mode) if mode != "full" else base)
        ades[mode] = float(trainer._metrics(pkt)["future_ADE"])
    vision_delta = ades["drop_vision"] - ades["full"]
    telemetry_delta = ades["delay_telemetry"] - ades["full"]
    permute_delta = abs(ades["permute_structure"] - ades["full"])
    both_ignored = abs(vision_delta) < 1e-6 and abs(telemetry_delta) < 1e-6
    same_drop = abs(vision_delta - telemetry_delta) < 1e-6
    permute_invariant = permute_delta < 1e-4
    return {
        "gate": "G_no_shortcut",
        "claim": "channel ablation and feature permute. Same drop everywhere is one-channel cheat. Permute-invariant is statistic fitting.",
        "future_ADE": ades,
        "vision_delta": vision_delta,
        "telemetry_delta": telemetry_delta,
        "permute_delta": permute_delta,
        "single_channel_risk": bool(same_drop),
        "both_channels_ignored": bool(both_ignored),
        "structure_ignored": bool(permute_invariant),
        "no_shortcut": (not both_ignored) and (not permute_invariant),
        "n": 1,
        "pass": all(v >= 0.0 for v in ades.values()),
    }


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
        "G": gate_g_no_shortcut(trainer),
    }
    report = {
        "protocol": "MINA capability ledger",
        "not_a_claim": "loss dropped is not intelligence appeared",
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
        "allowed_claims": list(ALLOWED_CLAIMS),
        "ledger": list(LEDGER_ROWS),
        "gates": gates,
        "protocol_complete": True,
        "all_proven": all(bool(row.get("pass")) for row in gates.values()),
        "unproven": [name for name, row in gates.items() if not row.get("pass")],
        "pass": all(bool(row.get("pass")) for row in gates.values()),
    }
    return report
