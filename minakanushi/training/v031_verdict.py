"""v0.3.1 H200 capability verdict. Not train-loss. Does not add layers.

Compares step128 vs step1128 on the sealed heldout pack:

    full heldout 100
    memory ON vs OFF ADE/FDE
    WAIT vs MOVE_TO on one world state
    revision slices
    action-path norms

6.8B construct is H200/B300 only.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from minakanushi.architecture.config import load_config, load_training
from minakanushi.architecture.freeze import is_6_8b_profile
from minakanushi.state.entity import AGENT_SLOT
from minakanushi.training.checkpoint import load_mina
from minakanushi.training.episode_dataset import JsonEpisodeDataset
from minakanushi.training.metrics import counterfactual_layers, counterfactual_separation_score
from minakanushi.training.trainer import Trainer, UnrollPacket
from minakanushi.training.v031_dataset import assert_v031_train_dataset

PHASES = ("physics", "agency", "causality", "embodiment")
MEMORY_SCENARIOS = frozenset(
    {
        "occlusion",
        "delayed",
        "sensor_delay",
        "motor_delay",
        "reacquisition",
        "hidden_object",
        "hidden_correction",
        "hidden_correction_l2",
        "hidden_correction_l3",
        "gone_forever",
    }
)
REVISION_SLICES = (
    "hidden_correction",
    "hidden_correction_l2",
    "hidden_correction_l3",
    "conflict",
    "wrong_velocity",
    "reacquisition",
    "unexpected_stop",
    "gone_forever",
)
SCALAR_KEYS = (
    "future_ADE",
    "future_FDE",
    "uncertainty_calibration_error",
    "revision_detected",
    "false_revision_rate",
    "revision_direction_accuracy",
)


def summarize(values: list[float]) -> dict[str, float]:
    """mean / median / p90 / worst-10. Empty list is not a PASS."""
    if not values:
        return {"n": 0.0, "mean": float("nan"), "median": float("nan"), "p90": float("nan"), "worst10": float("nan")}
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    mid = n // 2
    median = ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    p90 = ordered[min(n - 1, max(0, math.ceil(0.9 * n) - 1))]
    tail = max(1, math.ceil(0.1 * n))
    return {
        "n": float(n),
        "mean": sum(ordered) / n,
        "median": median,
        "p90": p90,
        "worst10": sum(ordered[-tail:]) / tail,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def intervention_deltas(pkt: UnrollPacket) -> dict[str, Any]:
    """Same world state: labeled future vs WAIT/MOVE_TO alternate."""
    labeled = pkt.pred_future[0]
    alt = pkt.alt_future[0]
    occ = pkt.aligned_occ[0].to(labeled.dtype).unsqueeze(-1)
    term = float(counterfactual_separation_score(labeled[-1], alt[-1]).detach())
    layers = counterfactual_layers(labeled[-1], alt[-1], pkt.aligned_occ[0])
    traj = float((((labeled - alt).pow(2) * occ.unsqueeze(0)).sum() / occ.sum().clamp_min(1.0) / labeled.shape[0]).sqrt().detach())
    rel_a = _pairwise(labeled[-1], pkt.aligned_occ[0])
    rel_b = _pairwise(alt[-1], pkt.aligned_occ[0])
    relation = float((rel_a - rel_b).abs().mean().detach())
    event = float((labeled[-1] - alt[-1]).abs().mean().detach())
    return {
        "terminal_delta": term,
        "occupied_terminal": float(layers["occupied_cf"].detach()),
        "agent_terminal": float(layers["agent_cf"].detach()),
        "frobenius_terminal": float(layers["frobenius"].detach()),
        "trajectory_delta": traj,
        "relation_delta": relation,
        "event_delta": event,
        "action_a": pkt.candidates[0].objective,
        "action_b": pkt.candidates[1].objective,
    }


def _pairwise(xy: Tensor, occ: Tensor) -> Tensor:
    mask = occ.bool()
    if int(mask.sum().item()) < 2:
        return torch.zeros((), device=xy.device, dtype=xy.dtype)
    d = torch.cdist(xy, xy)
    keep = mask.unsqueeze(0) & mask.unsqueeze(1)
    keep = keep & ~torch.eye(keep.shape[0], dtype=torch.bool, device=keep.device)
    if not bool(keep.any()):
        return torch.zeros((), device=xy.device, dtype=xy.dtype)
    return d[keep]


def action_trace(trainer: Trainer, pkt: UnrollPacket) -> dict[str, float]:
    """Where WAIT vs MOVE_TO dies: action vector or future residual."""
    engine = trainer.system.future
    world = pkt.pred
    agent = world.entity_xy[:, AGENT_SLOT]
    first, second = pkt.candidates[0], pkt.candidates[1]
    wait = first if first.objective == "WAIT" else second
    move = second if first.objective == "WAIT" else first
    vec_w = engine._action_vector(wait, agent)
    vec_m = engine._action_vector(move, agent)
    action_delta = float(torch.linalg.vector_norm(vec_w - vec_m).detach())
    fut = engine.predict(world, [wait, move], max_horizon=trainer.config.architecture.prediction_horizons.short)
    fw = next(t.states_xy for t in fut if t.strategy_id == wait.strategy_id)
    fm = next(t.states_xy for t in fut if t.strategy_id == move.strategy_id)
    future_delta = float(torch.linalg.vector_norm(fw[-1] - fm[-1]).mean().detach())
    layers = counterfactual_layers(fw[-1], fm[-1])
    return {
        "action_vector_delta": action_delta,
        "future_terminal_delta": future_delta,
        "official_cf": float(layers["official_cf"].detach()),
        "agent_cf": float(layers["agent_cf"].detach()),
        "frobenius": float(layers["frobenius"].detach()),
        "signal_reaches_future": action_delta > 1e-6,
        "future_uses_action": future_delta > 1e-3,
    }


def _row(trainer: Trainer, index: int, episode, phase: str) -> dict[str, Any]:
    with torch.no_grad():
        with trainer._amp():
            pkt = trainer.unroll(index + 1, episode=episode)
        metrics = trainer._metrics(pkt)
    inter = intervention_deltas(pkt)
    row = {
        "index": index,
        "phase": phase,
        "scenario": episode.scenario,
        "episode_index": int(episode.episode_index),
        "seed": int(getattr(episode, "seed", trainer.config.training.seed)),
        "split": "heldout",
        **{k: float(metrics[k]) for k in SCALAR_KEYS if k in metrics},
        "memory_ade_on": float(metrics["memory_ade_on"]),
        "memory_ade_off": float(metrics["memory_ade_off"]),
        "memory_fde_on": float(metrics["memory_fde_on"]),
        "memory_fde_off": float(metrics["memory_fde_off"]),
        "memory_helps": bool(metrics["memory_helps_future"]),
        "reacquisition_accuracy": float(metrics["reacquisition_accuracy"]),
        "revision_latency": float(metrics["revision_latency"]),
        "counterfactual_distance": float(metrics["counterfactual_quality"]),
        **inter,
    }
    return row


def evaluate_heldout(trainer: Trainer, held: JsonEpisodeDataset, *, traces: int = 20) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    traces_out: list[dict[str, Any]] = []
    trainer.system.eval()
    for i in range(len(held)):
        episode = held.episode(i)
        phase = held.phases[i]
        row = _row(trainer, i, episode, phase)
        rows.append(row)
        if len(traces_out) < traces:
            with torch.no_grad():
                pkt = trainer.unroll(i + 1, episode=episode)
                traces_out.append({"index": i, "scenario": episode.scenario, **action_trace(trainer, pkt)})
        print(
            f"verdict {i + 1}/{len(held)} {phase} {episode.scenario} "
            f"ADE={row['future_ADE']:.4f} mem_on={row['memory_ade_on']:.4f} "
            f"mem_off={row['memory_ade_off']:.4f} cf={row['terminal_delta']:.6f}",
            flush=True,
        )
    return {
        "n": len(rows),
        "aggregates": _aggregates(rows),
        "by_phase": {phase: _aggregates([r for r in rows if r["phase"] == phase]) for phase in PHASES},
        "memory": _memory_block(rows),
        "counterfactual": _cf_block(rows),
        "revision_slices": _revision_block(rows),
        "action_trace": {
            "n": len(traces_out),
            "mean_action_vector_delta": _mean([t["action_vector_delta"] for t in traces_out]),
            "mean_future_terminal_delta": _mean([t["future_terminal_delta"] for t in traces_out]),
            "action_reaches_future_rate": _mean([1.0 if t["signal_reaches_future"] else 0.0 for t in traces_out]),
            "future_uses_action_rate": _mean([1.0 if t["future_uses_action"] else 0.0 for t in traces_out]),
            "rows": traces_out,
        },
        "rows": rows,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _aggregates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {key: summarize([float(r[key]) for r in rows if key in r]) for key in SCALAR_KEYS}


def _memory_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    memory_rows = [r for r in rows if r["scenario"] in MEMORY_SCENARIOS]
    chosen = memory_rows or rows

    def _side(name: str) -> dict[str, Any]:
        return {
            "all_heldout": summarize([float(r[name]) for r in rows]),
            "memory_scenarios": summarize([float(r[name]) for r in memory_rows]),
        }

    on = [float(r["memory_ade_on"]) for r in chosen]
    off = [float(r["memory_ade_off"]) for r in chosen]
    helps = sum(1 for r in chosen if r["memory_ade_on"] < r["memory_ade_off"])
    return {
        "n_memory_scenarios": len(memory_rows),
        "ade_on": _side("memory_ade_on"),
        "ade_off": _side("memory_ade_off"),
        "fde_on": _side("memory_fde_on"),
        "fde_off": _side("memory_fde_off"),
        "help_rate": helps / max(len(chosen), 1),
        "pass": bool(on) and (_mean(on) < _mean(off)),
        "claim": "ADE(memory on) < ADE(memory off) on occlusion/delay/hidden slices. Not latent L2.",
    }


def _cf_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    term = [float(r["terminal_delta"]) for r in rows]
    occupied = [float(r["occupied_terminal"]) for r in rows if "occupied_terminal" in r]
    agent = [float(r["agent_terminal"]) for r in rows if "agent_terminal" in r]
    traj = [float(r["trajectory_delta"]) for r in rows]
    rel = [float(r["relation_delta"]) for r in rows]
    gate = occupied or term
    gated_on = "occupied" if occupied else "official"
    return {
        "terminal": summarize(term),
        "occupied": summarize(occupied) if occupied else None,
        "agent": summarize(agent) if agent else None,
        "trajectory": summarize(traj),
        "relation": summarize(rel),
        "std_terminal": _std(term),
        "std_occupied": _std(occupied) if occupied else None,
        "gated_on": gated_on,
        "existence": bool(gate) and max(gate) > 1e-4,
        "diversity": bool(gate) and _std(gate) > 1e-4,
        "pass_existence": bool(gate) and max(gate) > 1e-4,
        "pass_diversity": bool(gate) and _std(gate) > 1e-4,
        "official_diversity_failed": bool(term) and _std(term) <= 1e-4,
    }


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def _revision_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in REVISION_SLICES:
        slice_rows = [r for r in rows if r["scenario"] == name]
        if not slice_rows and name == "wrong_velocity":
            slice_rows = [r for r in rows if r["scenario"] == "hidden_correction_l2"]
        out[name] = {
            "n": len(slice_rows),
            "detection_rate": _mean([float(r["revision_detected"]) for r in slice_rows]),
            "direction_accuracy": _mean([float(r["revision_direction_accuracy"]) for r in slice_rows]),
            "false_revision_rate": _mean([float(r["false_revision_rate"]) for r in slice_rows]),
            "recovery_latency": _mean([float(r["revision_latency"]) for r in slice_rows]),
        }
    return out


def _clear_stale_torchrun() -> None:
    """Verdict is single-process. Leftover torchrun env must not wrap DTensors."""
    for key in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "GROUP_RANK",
        "ROLE_RANK",
        "LOCAL_WORLD_SIZE",
        "TORCHELASTIC_RUN_ID",
    ):
        os.environ.pop(key, None)


def eval_trainer(root: Path, training_yaml: Path, mina: Path) -> Trainer:
    """Single-process eval construct. Weights only. No torchrun. H200/B300 only for 6.8B."""
    _clear_stale_torchrun()
    training = load_training(training_yaml)
    assert_v031_train_dataset(root, training)
    config = load_config(
        root / training.architecture,
        training_path=training_yaml,
        runtime_path=root / "configs" / "runtime" / "cpu.yaml",
        simulation_path=root / training.simulation,
    )
    train = replace(config.training, parallelism="none", dataset_split="heldout", activation_checkpoint=False)
    trainer = Trainer(replace(config, training=train), root, eval_only=True)
    refuse_cpu_6_8b(trainer)
    print(f"construct ok device={trainer.device} loading {mina}", flush=True)
    load_mina(Path(mina), trainer.system, optimizer=None)
    trainer.system.eval()
    print(f"loaded {mina} eval_only dense", flush=True)
    return trainer


def compare_reports(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    held_b = before["aggregates"]["future_ADE"]["mean"]
    held_a = after["aggregates"]["future_ADE"]["mean"]
    rev_b = before["aggregates"]["revision_detected"]["mean"]
    rev_a = after["aggregates"]["revision_detected"]["mean"]
    false_b = before["aggregates"]["false_revision_rate"]["mean"]
    false_a = after["aggregates"]["false_revision_rate"]["mean"]
    dir_a = after["aggregates"]["revision_direction_accuracy"]["mean"]
    mem_pass = bool(after["memory"]["pass"])
    cf_exist = bool(after["counterfactual"]["pass_existence"])
    cf_div = bool(after["counterfactual"]["pass_diversity"])
    heldout_down = held_a < held_b
    revision_up = rev_a > rev_b
    false_ok = false_a <= max(false_b, 0.05)
    direction_fail = dir_a < 0.2
    signals_c = []
    if not mem_pass:
        signals_c.append("memory usefulness not demonstrated")
    if not cf_exist or not cf_div:
        signals_c.append("counterfactual separation failed")
    if direction_fail:
        signals_c.append("revision direction failed")
    if false_a > max(false_b + 0.05, 0.1):
        signals_c.append("false revision rose")
    variant_a = heldout_down and revision_up and mem_pass and false_ok and cf_exist and not direction_fail
    variant_c = (false_a > max(false_b + 0.05, 0.1)) or (rev_a + 1e-9 < rev_b and direction_fail)
    if variant_a:
        variant = "A"
    elif variant_c and not heldout_down:
        variant = "C"
    else:
        variant = "B"
    return {
        "variant": variant,
        "accepted": variant == "A",
        "heldout_ADE": {"before": held_b, "after": held_a, "improved": heldout_down},
        "revision_detected": {"before": rev_b, "after": rev_a, "improved": revision_up},
        "false_revision": {"before": false_b, "after": false_a, "ok": false_ok},
        "revision_direction_after": dir_a,
        "memory_pass": mem_pass,
        "counterfactual_existence": cf_exist,
        "counterfactual_diversity": cf_div,
        "c_signals": signals_c,
        "not_a_claim": "train loss is not this verdict. Single-episode eval is not this verdict.",
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def refuse_cpu_6_8b(trainer: Trainer) -> None:
    if is_6_8b_profile(trainer.config.architecture) and trainer.device.type == "cpu":
        raise RuntimeError("refusing 6.8B verdict on CPU. Use the H200.")
