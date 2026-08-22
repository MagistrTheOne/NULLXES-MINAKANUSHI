"""v0.3.1-R: sensor_delay revision cut and post-patch CPU forensic.

Does not train. Does not construct 6.8B.
Does not change REVISION_MAGNITUDE / MOVE_DETECT.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from minakanushi.architecture.mina_unit import KIND_IDS
from minakanushi.state.correction import REVISION_MAGNITUDE
from minakanushi.training.revision import (
    AGENT_ENTITY_ID,
    MOVE_DETECT,
    revision_metrics,
    should_revise_mask,
)
from simulations.synthetic_world.dataset import training_frame

SENSOR_DELAY_S = 0.15
MOTOR_DELAY_S = 0.30
DELAY_SCENARIOS = frozenset({"delayed", "sensor_delay", "motor_delay"})


def thresholds() -> dict[str, float]:
    return {
        "revision_magnitude": float(REVISION_MAGNITUDE),
        "move_detect": float(MOVE_DETECT),
        "sensor_delay_s": float(SENSOR_DELAY_S),
        "motor_delay_s": float(MOTOR_DELAY_S),
    }


def _mover_slot(truth) -> int:
    for i, kind in enumerate(truth.kind):
        if str(kind) == "mover":
            return int(i)
    raise ValueError(f"no mover in truth kinds={truth.kind}")


def _obs_xy(obs, entity_id: int) -> np.ndarray | None:
    for item in obs.visible:
        if int(item["id"]) == int(entity_id):
            return np.asarray(item["xy"], dtype=np.float64)
    return None


def episode_timing(episode) -> dict[str, Any]:
    idx = training_frame(episode.scenario, len(episode.observations))
    truth = episode.truth[idx]
    obs = episode.observations[idx]
    event = float(obs.timestamp)
    arrival = float(obs.arrival_time if obs.arrival_time is not None else obs.timestamp)
    return {
        "scenario": str(episode.scenario),
        "length": len(episode.observations),
        "training_frame": int(idx),
        "event_time": event,
        "arrival_time": arrival,
        "delay_s": arrival - event,
        "dt": float(episode.dt),
        "correction_frame_named": episode.scenario not in {"hidden_correction", "conflict", "gone_forever", "reacquisition"},
    }


def _frame_geometry(episode, idx: int) -> dict[str, Any]:
    if idx < 1:
        raise ValueError("need a previous frame for oracle-prev belief")
    now = episode.truth[idx]
    prev = episode.truth[idx - 1]
    obs = episode.observations[idx]
    slot = _mover_slot(now)
    eid = int(now.entity_id[slot])
    prev_hits = np.where(np.asarray(prev.entity_id) == eid)[0]
    prev_slot = int(prev_hits[0])
    truth_now = np.asarray(now.xy[slot], dtype=np.float64)
    truth_prev = np.asarray(prev.xy[prev_slot], dtype=np.float64)
    vel_now = np.asarray(now.vel[slot], dtype=np.float64)
    obs_xy = _obs_xy(obs, eid)
    visible = [(int(item["id"]), str(item["kind"])) for item in obs.visible]
    visible_movers = [row for row in visible if row[1] == "mover"]
    step = float(np.linalg.norm(truth_now - truth_prev))
    delay_s = float((obs.arrival_time if obs.arrival_time is not None else obs.timestamp) - obs.timestamp)
    delay_path = float(np.linalg.norm(vel_now) * delay_s)
    obs_vs_truth = float(np.linalg.norm(obs_xy - truth_now)) if obs_xy is not None else float("nan")
    visible_current = []
    for item in obs.visible:
        tid = int(item["id"])
        hits = np.where(np.asarray(now.entity_id) == tid)[0]
        if hits.size == 0:
            continue
        truth_xy = np.asarray(now.xy[int(hits[0])], dtype=np.float64)
        visible_current.append(float(np.linalg.norm(np.asarray(item["xy"], dtype=np.float64) - truth_xy)))
    return {
        "frame": int(idx),
        "entity_id": eid,
        "visible": visible,
        "n_visible_movers": len(visible_movers),
        "truth_step": step,
        "speed": float(np.linalg.norm(vel_now)),
        "delay_s": delay_s,
        "delay_path": delay_path,
        "obs_vs_truth_now": obs_vs_truth,
        "mover_in_observation": obs_xy is not None,
        "visible_xy_is_current": bool(visible_current) and max(visible_current) < 0.5,
        "teacher_if_belief_is_truth_prev": step >= REVISION_MAGNITUDE,
        "teacher_if_belief_vs_delay_stale": delay_path >= REVISION_MAGNITUDE,
        "below_revision_magnitude": {
            "one_step": step < REVISION_MAGNITUDE,
            "delay_path": delay_path < REVISION_MAGNITUDE,
        },
    }


def last_mover_visible_frame(episode) -> int:
    last = -1
    for t, obs in enumerate(episode.observations):
        if any(str(item.get("kind")) == "mover" for item in obs.visible):
            last = t
    return last


def spatial_disagreement(episode) -> dict[str, Any]:
    """Geometry at the trained transition vs the frame that still sees a mover."""
    train_idx = training_frame(episode.scenario, len(episode.observations))
    early_idx = 2 if len(episode.observations) > 3 else 1
    trained = _frame_geometry(episode, train_idx)
    early = _frame_geometry(episode, early_idx)
    last_vis = last_mover_visible_frame(episode)
    return {
        **trained,
        "training_frame": int(train_idx),
        "early_delay_frame": int(early_idx),
        "early": early,
        "last_mover_visible_frame": last_vis,
        "train_frame_after_mover_left": last_vis >= 0 and train_idx > last_vis,
        "obs_is_current_xy": bool(early["visible_xy_is_current"]),
        "teacher_if_belief_is_truth_now": False,
        "constructor_path": "tracking" if early["n_visible_movers"] else "no_mover_evidence",
        "constructor_note": (
            "At the early delay frame the mover is visible and consecutive, so "
            "revise_slot uses tracking. The trained frame is length//2; the mover "
            "has usually left sensor range, so the teacher sees only static evidence."
        ),
        "cut": (
            "train_frame_has_no_mover_evidence"
            if trained["n_visible_movers"] == 0
            else "teacher_below_threshold"
            if not trained["teacher_if_belief_is_truth_prev"]
            else "teacher_would_fire"
        ),
    }


def metric_empty_teacher_is_not_a_miss() -> dict[str, Any]:
    """n_need==0 is excluded from the detection denominator."""
    before = torch.zeros(1, 2, 2)
    evidence = torch.zeros(1, 2, 2)
    after = torch.zeros(1, 2, 2)
    has = torch.tensor([[True, True]])
    occ = torch.tensor([[True, True]])
    ids = torch.tensor([[11, 1]])
    should = should_revise_mask(before, evidence, has, occ, ids)
    metrics = revision_metrics(
        before_xy=before,
        after_xy=after,
        evidence_xy=evidence,
        should_revise=should,
        has_evidence=has,
        occupied_before=occ,
        entity_id=ids,
    )
    detected = float(metrics["revision_detected"])
    return {
        "n_need": int(metrics["n_need"]),
        "n_detected": int(metrics["n_detected"]),
        "n_no_need": int(metrics["n_no_need"]),
        "n_false_revision": int(metrics["n_false_revision"]),
        "revision_detected": detected,
        "excluded_from_detection": int(should.sum().item()) == 0 and math.isnan(detected),
        "false_revision_rate": float(metrics["false_revision_rate"]),
        "claim": "empty teacher is not a missed revision. Only false_revision is scored.",
    }


def classify_cut(
    *,
    n_need: int,
    max_before_d: float,
    detected: float,
    n_mover_evidence: int | None = None,
    magnitude: float = REVISION_MAGNITUDE,
) -> str:
    if n_mover_evidence is not None and int(n_mover_evidence) == 0:
        return "no_mover_evidence"
    if int(n_need) == 0 and float(max_before_d) < float(magnitude):
        return "teacher_suppressed"
    if int(n_need) > 0 and not math.isnan(float(detected)) and float(detected) <= 0.0:
        return "model_did_not_move"
    if int(n_need) > 0 and float(detected) > 0.0:
        return "trigger_live"
    return "teacher_empty_other"


def live_slot_audit(pkt) -> dict[str, Any]:
    """Read UnrollPacket slot residuals. Caller already ran unroll."""
    before = pkt.before_xy[0]
    evidence = pkt.evidence_xy[0]
    after = pkt.pred.entity_xy[0]
    ids = pkt.pred.entity_id[0]
    has = pkt.has_evidence[0]
    occ = pkt.occupied_before[0]
    should = pkt.should_revise[0]
    before_d = torch.linalg.vector_norm(before - evidence, dim=-1)
    moved = torch.linalg.vector_norm(after - before, dim=-1)
    after_d = torch.linalg.vector_norm(after - evidence, dim=-1)
    not_self = ids != AGENT_ENTITY_ID
    watch = has & occ & not_self
    n_need = int(should.sum().item())
    n_watch = int(watch.sum().item())
    max_before = float(before_d[watch].max().item()) if n_watch else 0.0
    mean_before = float(before_d[watch].mean().item()) if n_watch else 0.0
    mean_moved = float(moved[watch].mean().item()) if n_watch else 0.0
    metrics = revision_metrics(
        before_xy=pkt.before_xy,
        after_xy=pkt.pred.entity_xy,
        evidence_xy=pkt.evidence_xy,
        should_revise=pkt.should_revise,
        has_evidence=pkt.has_evidence,
        occupied_before=pkt.occupied_before,
        entity_id=pkt.pred.entity_id,
    )
    xy_std = None
    conflict = None
    if hasattr(pkt.pred, "xy_std") and n_watch:
        xy_std = float(torch.linalg.vector_norm(pkt.pred.xy_std[0][watch], dim=-1).mean().item())
    if hasattr(pkt.pred, "uncertainty") and n_watch:
        conflict = float(pkt.pred.uncertainty[0][watch, 2].mean().item())
    n_mover_evidence = 0
    if hasattr(pkt.pred, "kind"):
        mover = pkt.pred.kind[0] == KIND_IDS["mover"]
        n_mover_evidence = int((has & occ & mover).sum().item())
    cut = classify_cut(
        n_need=n_need,
        max_before_d=max_before,
        detected=float(metrics["revision_detected"]),
        n_mover_evidence=n_mover_evidence,
    )
    return {
        "scenario": str(pkt.scenario),
        "frame_index": int(pkt.frame_index),
        "n_watch": n_watch,
        "n_mover_evidence": n_mover_evidence,
        "n_need": n_need,
        "max_before_d": max_before,
        "mean_before_d": mean_before,
        "mean_moved": mean_moved,
        "mean_after_d": float(after_d[watch].mean().item()) if n_watch else 0.0,
        "mean_xy_std": xy_std,
        "mean_conflict": conflict,
        "n_constructor_corrections": int(pkt.n_constructor_corrections),
        "revision_detected": float(metrics["revision_detected"]),
        "revision_required_recall": float(metrics["revision_required_recall"]),
        "revision_direction_accuracy": float(metrics["revision_direction_accuracy"]),
        "false_revision_rate": float(metrics["false_revision_rate"]),
        "n_need_slots": int(metrics["n_need"]),
        "n_detected": int(metrics["n_detected"]),
        "n_no_need": int(metrics["n_no_need"]),
        "n_false_revision": int(metrics["n_false_revision"]),
        "cut": cut,
        "teacher_below_threshold": n_need == 0 and max_before < REVISION_MAGNITUDE,
        "better_prediction_suppresses_teacher": (
            n_need == 0 and max_before < REVISION_MAGNITUDE and n_mover_evidence > 0
        ),
    }


def _verdict_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a rows list")
    return rows


def compare_sensor_delay_verdicts(before: Path, after: Path) -> dict[str, Any]:
    def _slice(rows: list[dict[str, Any]]) -> dict[str, Any]:
        group = [r for r in rows if str(r.get("scenario")) == "sensor_delay"]
        if not group:
            return {"n": 0}
        ade = [float(r["future_ADE"]) for r in group]
        det = [
            float(r["revision_detected"])
            for r in group
            if not (isinstance(r.get("revision_detected"), float) and math.isnan(float(r["revision_detected"])))
        ]
        return {
            "n": len(group),
            "ade_mean": sum(ade) / len(ade),
            "detection_mean": sum(det) / len(det) if det else float("nan"),
            "detection_n_scored": len(det),
            "detection_all_zero": bool(det) and all(v == 0.0 for v in det),
        }

    b = _slice(_verdict_rows(before))
    a = _slice(_verdict_rows(after))
    return {
        "before": b,
        "after": a,
        "ade_improved": bool(b.get("n") and a.get("n") and a["ade_mean"] < b["ade_mean"]),
        "detection_dropped": bool(b.get("n") and a.get("n") and a["detection_mean"] < b["detection_mean"]),
        "hypothesis": (
            "The trained sensor_delay frame usually has no mover. Remaining evidence "
            "is static. step1128 tracks it inside 0.25 so the teacher never asks "
            "for revision. step128 residual on that same obstacle accidentally fired."
        ),
    }


def dataset_sensor_delay_geometry(dataset: Path, *, split: str = "heldout") -> dict[str, Any]:
    from minakanushi.training.episode_dataset import JsonEpisodeDataset

    held = JsonEpisodeDataset(dataset, seed=11, split=split)
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(held.scenarios):
        if name not in DELAY_SCENARIOS:
            continue
        episode = held.episode(i)
        rows.append({**episode_timing(episode), **spatial_disagreement(episode), "phase": held.phases[i]})
    sensor = [r for r in rows if r["scenario"] == "sensor_delay"]
    return {
        "n_delay_episodes": len(rows),
        "n_sensor_delay": len(sensor),
        "sensor_delay_mean_step": (
            sum(r["truth_step"] for r in sensor) / len(sensor) if sensor else float("nan")
        ),
        "sensor_delay_mean_delay_path": (
            sum(r["delay_path"] for r in sensor) / len(sensor) if sensor else float("nan")
        ),
        "sensor_delay_teacher_if_oracle_prev_rate": (
            sum(1.0 for r in sensor if r["teacher_if_belief_is_truth_prev"]) / len(sensor) if sensor else float("nan")
        ),
        "sensor_delay_train_frame_no_mover_rate": (
            sum(1.0 for r in sensor if r["n_visible_movers"] == 0) / len(sensor) if sensor else float("nan")
        ),
        "sensor_delay_obs_is_current_xy_rate": (
            sum(1.0 for r in sensor if r["obs_is_current_xy"]) / len(sensor) if sensor else float("nan")
        ),
        "rows": rows,
    }


def _cpu_simulation():
    from minakanushi.architecture.config import load_config

    root = Path(__file__).resolve().parents[2]
    return load_config(
        root / "configs" / "architecture" / "cpu_dev.yaml",
        runtime_path=root / "configs" / "runtime" / "cpu.yaml",
        simulation_path=root / "configs" / "simulation" / "milestone1.yaml",
    ).simulation


def generated_sensor_delay_geometry(*, seeds: tuple[int, ...] = (11, 17, 23), length: int = 32) -> dict[str, Any]:
    from simulations.synthetic_world.dataset import generate_episode

    sim = _cpu_simulation()
    rows = []
    for seed in seeds:
        episode = generate_episode(sim, seed=seed, episode_index=0, length=length, scenario="sensor_delay")
        rows.append({**episode_timing(episode), **spatial_disagreement(episode)})
    return {
        "n": len(rows),
        "source": "generate_episode",
        "length": length,
        "mean_step": sum(r["truth_step"] for r in rows) / len(rows),
        "mean_delay_path": sum(r["delay_path"] for r in rows) / len(rows),
        "teacher_if_oracle_prev_rate": sum(1.0 for r in rows if r["teacher_if_belief_is_truth_prev"]) / len(rows),
        "obs_is_current_xy_rate": sum(1.0 for r in rows if r["obs_is_current_xy"]) / len(rows),
        "train_frame_no_mover_rate": sum(1.0 for r in rows if r["n_visible_movers"] == 0) / len(rows),
        "early_mover_visible_rate": sum(1.0 for r in rows if r["early"]["n_visible_movers"] > 0) / len(rows),
        "early_teacher_if_oracle_prev_rate": (
            sum(1.0 for r in rows if r["early"]["teacher_if_belief_is_truth_prev"]) / len(rows)
        ),
        "rows": rows,
    }


def diagnose(
    *,
    dataset: Path | None = None,
    before_verdict: Path | None = None,
    after_verdict: Path | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "cycle": "v0.3.1-R",
        "title": "Revision Trigger Diagnostic",
        "trains": False,
        "constructs_6_8b": False,
        "thresholds": thresholds(),
        "metric_identity": metric_empty_teacher_is_not_a_miss(),
        "generated": generated_sensor_delay_geometry(),
        "cut_points": [
            "A. sensor_delay trains/evals curriculum frame 2, not length//2.",
            "B. n_need==0 is excluded from revision_required_recall.",
            "C. delay teacher is mover-only; leftover obstacle residual is not a target.",
            "D. REVISION_MAGNITUDE stays 0.25. Weights stay step1128.mina.",
        ],
    }
    if dataset is not None:
        report["heldout_geometry"] = dataset_sensor_delay_geometry(dataset)
    else:
        report["heldout_geometry"] = {"status": "skipped", "reason": "no --dataset"}
    if before_verdict is not None and after_verdict is not None:
        report["verdict_compare"] = compare_sensor_delay_verdicts(before_verdict, after_verdict)
    else:
        report["verdict_compare"] = {"status": "skipped", "reason": "need --before-verdict and --after-verdict"}
    gen = report["generated"]
    ident = report["metric_identity"]
    frames = [int(r["training_frame"]) for r in gen["rows"]]
    report["cpu_verdict"] = {
        "sensor_delay_frame": frames[0] if frames else None,
        "sensor_delay_frame_not_mid": all(idx != 32 and idx != length // 2 for idx, length in ((r["training_frame"], r["length"]) for r in gen["rows"])),
        "mover_evidence_at_train_frame": gen["train_frame_no_mover_rate"] == 0.0,
        "empty_teacher_not_missed_detection": bool(ident["excluded_from_detection"]),
        "delay_is_timestamp_only": gen["obs_is_current_xy_rate"] == 1.0,
        "early_frame_has_mover": gen["early_mover_visible_rate"] == 1.0,
        "accepted": False,
        "variant": "B",
        "next": "CPU patch PASS → H200 live_after_patch on the same step1128.mina. Do not train.",
    }
    report["cpu_verdict"]["cpu_patch_pass"] = bool(
        report["cpu_verdict"]["sensor_delay_frame_not_mid"]
        and report["cpu_verdict"]["mover_evidence_at_train_frame"]
        and report["cpu_verdict"]["empty_teacher_not_missed_detection"]
    )
    return report
