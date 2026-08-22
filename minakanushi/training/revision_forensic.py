"""v0.3.1-R: where sensor_delay revision is cut.

Does not train. Does not construct 6.8B.
Does not change REVISION_MAGNITUDE / MOVE_DETECT.

Teacher is geometric: |belief − evidence| >= 0.25 on a non-self slot.
Detection is empty-teacher collapse: n_need==0 → revision_detected==0.0.
sensor_delay stamps arrival_time; observe() still emits current xy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from minakanushi.architecture.mina_unit import KIND_IDS
from minakanushi.state.correction import REVISION_MAGNITUDE
from minakanushi.training.revision import AGENT_ENTITY_ID, MOVE_DETECT, revision_metrics, should_revise_mask
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


def metric_collapses_when_teacher_empty() -> dict[str, Any]:
    """Identity of revision_metrics: n_need==0 reports detected=0.0."""
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
    return {
        "n_need": int(should.sum().item()),
        "revision_detected": float(metrics["revision_detected"]),
        "collapses_to_zero": int(should.sum().item()) == 0 and metrics["revision_detected"] == 0.0,
        "claim": "empty teacher is scored as missed detection, not as 'no revision needed'.",
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
    if int(n_need) > 0 and float(detected) <= 0.0:
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
        "revision_direction_accuracy": float(metrics["revision_direction_accuracy"]),
        "false_revision_rate": float(metrics["false_revision_rate"]),
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
        det = [float(r["revision_detected"]) for r in group]
        return {
            "n": len(group),
            "ade_mean": sum(ade) / len(ade),
            "detection_mean": sum(det) / len(det),
            "detection_all_zero": all(v == 0.0 for v in det),
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
        "metric_identity": metric_collapses_when_teacher_empty(),
        "generated": generated_sensor_delay_geometry(),
        "cut_points": [
            "1. sensor_delay delay is arrival_time stamp, not a stale xy.",
            "2. trainer unrolls length//2; the mover has already left sensor range.",
            "3. should_revise ignores curriculum rows (frame 2); only |Δxy|>=0.25.",
            "4. consecutive visibility is tracking, not hypothesis_revision.",
            "5. revision_metrics scores n_need==0 as detected==0.",
            "6. a better tracker on leftover static evidence kills the teacher.",
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
    report["cpu_verdict"] = {
        "delay_is_timestamp_only": gen["obs_is_current_xy_rate"] == 1.0,
        "train_frame_has_no_mover": gen["train_frame_no_mover_rate"] == 1.0,
        "early_frame_has_mover": gen["early_mover_visible_rate"] == 1.0,
        "oracle_prev_teacher_dead": gen["teacher_if_oracle_prev_rate"] == 0.0,
        "early_oracle_prev_teacher_dead": gen["early_teacher_if_oracle_prev_rate"] == 0.0,
        "one_step_below_threshold": gen["mean_step"] < REVISION_MAGNITUDE,
        "delay_path_below_threshold": gen["mean_delay_path"] < REVISION_MAGNITUDE,
        "next": (
            "H200 live dump on sensor_delay: n_mover_evidence and max_before_d, "
            "step128 vs step1128. Expect train frame n_mover_evidence=0. "
            "If 1128 max_before_d < 0.25 on leftover static slots, the remaining "
            "defect is teacher/frame calibration, not a missing Future Engine."
        ),
    }
    return report
