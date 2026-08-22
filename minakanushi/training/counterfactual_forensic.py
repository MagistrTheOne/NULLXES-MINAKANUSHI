"""v0.3.2 diagnostic: where official cf turns 0.40 into 0.000786.

Does not construct 6.8B. Does not train.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from minakanushi.state.entity import AGENT_SLOT
from minakanushi.training.metrics import counterfactual_layers, counterfactual_separation_score
from minakanushi.training.phase_sampler import PhaseCurriculumSampler, mode_for_job_step
from minakanushi.training.v031_verdict import _mean, _std, summarize

# Measured on H200 heldout-100, step1128. Not a new claim of intelligence.
V031_OFFICIAL_CF_MEAN = 0.0007857031049206853
V031_OFFICIAL_CF_STD = 3.3952208637715273e-06
V031_FUTURE_FROBENIUS = 0.40054136961698533
V031_WORLD_SLOTS = 512
# Trainer: after resume step128, start_step=129, end=start+1000-1=1128.
# mode uses job = step - start_step + 1; choose() RNG uses global step.
V031_RESUME_START = 129
V031_LAST_STEP = 1128
V031_WARM_STEPS = 16
MEMORY_EMBODIMENT = frozenset({"sensor_delay", "motor_delay", "agent_move", "gone_forever"})
SAMPLER_FOCUS = (
    "gone_forever",
    "sensor_delay",
    "motor_delay",
    "hidden_correction",
    "hidden_correction_l2",
    "hidden_correction_l3",
    "conflict",
    "reacquisition",
    "unexpected_stop",
    "hidden_object",
)


def metric_collapse_report(*, slots: int = V031_WORLD_SLOTS, agent_delta: float = 0.40) -> dict[str, Any]:
    """CPU identity: official_cf = agent_L2 / n_slots when other slots are zero."""
    future_a = torch.zeros(slots, 2)
    future_b = torch.zeros(slots, 2)
    future_b[AGENT_SLOT, 0] = float(agent_delta)
    occupied = torch.zeros(slots)
    occupied[AGENT_SLOT] = 1.0
    layers = {key: float(value.detach()) for key, value in counterfactual_layers(future_a, future_b, occupied).items()}
    official = float(counterfactual_separation_score(future_a, future_b).detach())
    recovered = V031_OFFICIAL_CF_MEAN * float(slots)
    recovered_std = V031_OFFICIAL_CF_STD * float(slots)
    return {
        "claim": "official cf is mean L2 over every world slot, including empty ones.",
        "synthetic": layers,
        "synthetic_official": official,
        "expected_official": float(agent_delta) / float(slots),
        "matches_identity": abs(official - float(agent_delta) / float(slots)) < 1e-6,
        "v031_ledger": {
            "official_cf_mean": V031_OFFICIAL_CF_MEAN,
            "official_cf_std": V031_OFFICIAL_CF_STD,
            "future_engine_frobenius": V031_FUTURE_FROBENIUS,
            "recovered_agent_mean": recovered,
            "recovered_agent_std": recovered_std,
            "frobenius_over_official": V031_FUTURE_FROBENIUS / V031_OFFICIAL_CF_MEAN,
            "slots": float(slots),
            "identity_holds": abs(recovered - V031_FUTURE_FROBENIUS) < 0.02,
            "recovered_diversity_would_pass": recovered_std > 1e-4,
        },
        "fork": "A" if abs(recovered - V031_FUTURE_FROBENIUS) < 0.02 else "C",
        "next": (
            "fix acceptance metric to occupied/agent cf, rerun heldout-100, no train"
            if abs(recovered - V031_FUTURE_FROBENIUS) < 0.02
            else "signal dies before terminal; inspect Future Engine"
        ),
    }


def replay_sampler(
    phases: tuple[str, ...],
    scenarios: tuple[str, ...],
    *,
    seed: int = 11,
    first_step: int = V031_RESUME_START,
    last_step: int = V031_LAST_STEP,
    resume_start: int = V031_RESUME_START,
    warm_steps: int = V031_WARM_STEPS,
) -> dict[str, Any]:
    """Replay the v0.3.1 optimizer order. Not dataset JSON counts.

    Trainer keys `choose(step)` on the global step. Warm/intelligence uses
    `job = step - start_step + 1`. After resume from step128 that is 129..1128.
    """
    if len(phases) != len(scenarios):
        raise ValueError("phases / scenarios length mismatch")
    if last_step < first_step:
        raise ValueError(f"last_step {last_step} < first_step {first_step}")
    dummy = tuple(Path(f"{i}.json") for i in range(len(phases)))
    sampler = PhaseCurriculumSampler(dummy, phases, seed=seed)
    phase_hits: Counter[str] = Counter()
    scenario_hits: Counter[str] = Counter()
    mode_hits: Counter[str] = Counter()
    steps_out: list[dict[str, Any]] = []
    for step in range(int(first_step), int(last_step) + 1):
        job = int(step) - int(resume_start) + 1
        mode = mode_for_job_step(job, warm_steps=warm_steps)
        idx = sampler.choose(step, mode)
        phase_hits[phases[idx]] += 1
        scenario_hits[scenarios[idx]] += 1
        mode_hits[mode] += 1
        steps_out.append(
            {
                "step": int(step),
                "job": int(job),
                "mode": mode,
                "phase": phases[idx],
                "scenario": scenarios[idx],
                "paired_wait_move": True,
            }
        )
    n_steps = last_step - first_step + 1
    focus = {name: int(scenario_hits.get(name, 0)) for name in SAMPLER_FOCUS}
    return {
        "first_step": int(first_step),
        "last_step": int(last_step),
        "resume_start": int(resume_start),
        "steps": n_steps,
        "seed": seed,
        "warm_steps": warm_steps,
        "n_train_episodes": len(phases),
        "modes": dict(mode_hits),
        "phases": dict(phase_hits),
        "scenarios": dict(scenario_hits),
        "focus_scenarios": focus,
        "paired_wait_move_every_step": n_steps,
        "optimizer_steps": steps_out,
        "note": (
            "choose(step) uses global step 129..1128. mode uses job 1..1000. "
            "Every train step already builds WAIT vs MOVE_TO via counterfactual_candidate. "
            "L_action still dilutes that pair by empty slots."
        ),
    }


def embodiment_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Detection regression is local, not global."""
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(str(row.get("scenario")), []).append(row)
    embodiment = [r for r in rows if r.get("phase") == "embodiment"]
    slices: dict[str, Any] = {}
    for name, group in sorted(by_scenario.items()):
        if name not in MEMORY_EMBODIMENT and name not in SAMPLER_FOCUS and not any(
            r.get("phase") == "embodiment" for r in group
        ):
            continue
        slices[name] = {
            "n": len(group),
            "phase": group[0].get("phase"),
            "detection": _mean([r.get("revision_detected") for r in group]),
            "direction": _mean([r.get("revision_direction_accuracy") for r in group]),
            "false_revision": _mean([float(r.get("false_revision_rate", 0.0)) for r in group]),
            "ade": _mean([float(r.get("future_ADE", float("nan"))) for r in group if "future_ADE" in r]),
        }
    return {
        "embodiment_n": len(embodiment),
        "embodiment_detection": _mean([r.get("revision_detected") for r in embodiment]),
        "sensor_delay": slices.get("sensor_delay"),
        "agent_move": slices.get("agent_move"),
        "slices": slices,
        "gone_forever_n_is_small": int(slices.get("gone_forever", {}).get("n", 0)) <= 3,
        "do_not_change_global_objective": True,
    }


def layers_from_packet(pred_future, alt_future, occupied) -> dict[str, float]:
    labeled = pred_future[0] if pred_future.ndim == 4 else pred_future
    alt = alt_future[0] if alt_future.ndim == 4 else alt_future
    occ = occupied[0] if occupied.ndim == 2 else occupied
    raw = counterfactual_layers(labeled[-1], alt[-1], occ, agent_slot=AGENT_SLOT)
    return {key: float(value.detach()) for key, value in raw.items()}


def load_index_rows(dataset: Path, *, split: str = "train") -> tuple[tuple[str, ...], tuple[str, ...]]:
    index = Path(dataset) / split / "index.jsonl"
    if not index.is_file():
        raise FileNotFoundError(index)
    phases: list[str] = []
    scenarios: list[str] = []
    for line in index.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        phases.append(str(rec.get("phase") or Path(rec["path"]).parent.name))
        scenarios.append(str(rec["scenario"]))
    return tuple(phases), tuple(scenarios)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): _clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_clean(v) for v in value]
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value

    path.write_text(json.dumps(_clean(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def diagnose(
    *,
    dataset: Path | None = None,
    verdict_rows: Path | None = None,
    slots: int = V031_WORLD_SLOTS,
    first_step: int = V031_RESUME_START,
    last_step: int = V031_LAST_STEP,
    resume_start: int = V031_RESUME_START,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "cycle": "v0.3.2-diagnostic",
        "trains": False,
        "constructs_6_8b": False,
        "metric": metric_collapse_report(slots=slots),
        "aggregates_note": summarize([V031_OFFICIAL_CF_MEAN]),
        "official_std": _std([V031_OFFICIAL_CF_MEAN, V031_OFFICIAL_CF_MEAN + V031_OFFICIAL_CF_STD]),
    }
    if dataset is not None:
        phases, scenarios = load_index_rows(dataset)
        report["sampler"] = replay_sampler(
            phases,
            scenarios,
            first_step=first_step,
            last_step=last_step,
            resume_start=resume_start,
        )
    else:
        report["sampler"] = {"status": "skipped", "reason": "no --dataset"}
    if verdict_rows is not None:
        payload = json.loads(Path(verdict_rows).read_text(encoding="utf-8"))
        rows = payload.get("rows") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("verdict JSON must contain a rows list")
        report["embodiment"] = embodiment_audit(rows)
    else:
        report["embodiment"] = {"status": "skipped", "reason": "no --verdict"}
    return report
