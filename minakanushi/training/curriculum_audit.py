"""Extended v0.3 curriculum audit. Distribution, action balance, diversity spread."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

from minakanushi.training.heldout import SKIP_DIR_NAMES, SKIP_JSON_NAMES, load_pack_index
from simulations.synthetic_world.curriculum_6_8b import PHASE_LENGTHS
from simulations.synthetic_world.dataset_v1 import REQUIRED_6_8B_KEYS, validate_episode_record

V03_MIN_EPISODES = 1000
V03_MIN_CORRECTIONS = 2000
MAX_ACTION_FRACTION = 0.95
REQUIRED_ACTIONS = ("WAIT", "MOVE_TO", "FOLLOW", "AVOID")
AGENCY_SCENARIOS = frozenset({"agent_move", "follow", "avoid", "goal_change"})
# User-facing names. goal_change is stored as agent_changes_goal on correction rows.
REQUIRED_REVISION = {
    "wrong_velocity": ("wrong_velocity",),
    "wrong_intent": ("wrong_intent",),
    "hidden_object": ("hidden_object",),
    "goal_change": ("agent_changes_goal", "goal_change"),
    "unexpected_physics": ("unexpected_physics",),
    "sensor_delay": ("sensor_delay",),
}


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, math.sqrt(var)


def _episode_paths(root: Path, *, split: str = "") -> list[Path]:
    root = Path(root)
    if split:
        rows = []
        index = root / split / "index.jsonl"
        if not index.is_file():
            raise FileNotFoundError(index)
        import json

        for line in index.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                rows.append(root / rec["path"])
        return rows
    indexed = load_pack_index(root)
    if indexed:
        return [root / row["path"] for row in indexed]
    found = []
    for path in sorted(root.rglob("*.json")):
        if path.name in SKIP_JSON_NAMES or path.parent.name in SKIP_DIR_NAMES:
            continue
        found.append(path)
    return found


def audit_curriculum(root: Path, *, gate: bool = False, split: str = "") -> dict[str, Any]:
    import json

    records = []
    for path in _episode_paths(root, split=split):
        rec = json.loads(path.read_text(encoding="utf-8"))
        rec["_path"] = str(path)
        validate_episode_record(rec, curriculum_6_8b=True)
        records.append(rec)
    phase_counts = Counter(str(r.get("phase", "missing")) for r in records)
    scenario_counts = Counter(str(r.get("scenario", "missing")) for r in records)
    lengths = sorted({len(r.get("transitions", [])) for r in records})
    obs_lengths = sorted({len(r.get("observations", [])) for r in records})
    correction_count = sum(len(r.get("corrections", [])) for r in records)
    diversities = [float(r.get("future_diversity") or 0.0) for r in records]
    div_mean, div_std = _mean_std(diversities)
    pwm = any(bool((r.get("embodiment") or {}).get("pwm")) for r in records)
    typed: Counter[str] = Counter()
    for rec in records:
        for row in rec.get("corrections") or []:
            typed[str(row.get("correction_type") or row.get("lesson") or "untagged")] += 1
    revision_counts = {}
    for name, aliases in REQUIRED_REVISION.items():
        revision_counts[name] = int(sum(typed[alias] for alias in aliases))
    actions: Counter[str] = Counter()
    for rec in records:
        for act in rec.get("actions") or []:
            actions[str(act.get("objective", "NONE"))] += 1
    action_total = sum(actions.values()) or 1
    action_fractions = {name: actions[name] / action_total for name in actions}
    max_action_fraction = max(action_fractions.values()) if action_fractions else 0.0
    wait_required = 0
    wait_safe_button = 0
    for rec in records:
        scenario = str(rec.get("scenario", ""))
        for act in rec.get("actions") or []:
            if str(act.get("objective")) != "WAIT":
                continue
            if scenario in AGENCY_SCENARIOS:
                wait_safe_button += 1
            else:
                wait_required += 1
    wait_total = wait_required + wait_safe_button
    entropy = 0.0
    for frac in action_fractions.values():
        if frac > 0.0:
            entropy -= frac * math.log2(frac)
    production = len(records) >= V03_MIN_EPISODES
    revision_ok = (not production) or all(count > 0 for count in revision_counts.values())
    action_ok = (not production) or (
        all(actions[name] > 0 for name in REQUIRED_ACTIONS) and max_action_fraction < MAX_ACTION_FRACTION
    )
    if not records:
        spread_ok = False
        collapsed = False
    else:
        collapsed = (max(diversities) - min(diversities)) < 1e-9
        spread_ok = min(diversities) > 1e-6
        if not production:
            spread_ok = True
    report = {
        "dataset_root": str(root),
        "split": split or "all",
        "n_episodes": len(records),
        "phase_counts": dict(phase_counts),
        "scenario_counts": dict(scenario_counts),
        "transition_lengths": lengths,
        "observation_lengths": obs_lengths,
        "correction_count": correction_count,
        "correction_types": dict(typed),
        "revision_distribution": revision_counts,
        "action_counts": dict(actions),
        "action_fractions": action_fractions,
        "max_action_fraction": max_action_fraction,
        "decision_entropy": entropy,
        "wait_required": wait_required,
        "wait_safe_button": wait_safe_button,
        "wait_required_fraction": wait_required / max(wait_total, 1),
        "event_count": sum(len(r.get("events", [])) for r in records),
        "future_diversity_mean": div_mean,
        "future_diversity_std": div_std,
        "future_diversity_min": min(diversities) if diversities else 0.0,
        "future_diversity_max": max(diversities) if diversities else 0.0,
        "future_diversity_positive": sum(1 for d in diversities if d > 1e-6),
        "future_diversity_collapsed": collapsed,
        "required_keys": list(REQUIRED_6_8B_KEYS),
        "pwm": pwm,
        "phase_lengths_contract": dict(PHASE_LENGTHS),
        "source_of_truth": "dataset/mina_6_8b_v03",
        "hf_role": "adapter_only",
        "gate": {
            "n_episodes": len(records) >= V03_MIN_EPISODES,
            "correction_density": correction_count >= V03_MIN_CORRECTIONS,
            "future_diversity": all(d > 1e-6 for d in diversities) if records else False,
            "pwm_false": pwm is False,
            "observation_spans": set(obs_lengths).issubset({32, 64}) if obs_lengths else False,
            "revision_types_nonzero": revision_ok,
            "action_balance": action_ok,
            "future_diversity_spread": spread_ok if records else False,
        },
        "warnings": {
            "future_diversity_collapsed": collapsed,
            "note": "min/max/mean/std are recorded. collapse=true means fork distance is nearly constant across episodes (arena geometry), not that forks are missing. Existence PASS, diversity NOT PASS. v0.4 geometry expansion.",
            "wait_note": "WAIT majority is observe-default, not MOVE_TO 95%. wait_safe_button>0 means WAIT on agency scenarios that should MOVE/FOLLOW/AVOID.",
        },
    }
    if gate:
        failed = [name for name, ok in report["gate"].items() if not ok]
        if failed:
            raise SystemExit(f"v0.3 curriculum gate failed: {failed}")
    return report
