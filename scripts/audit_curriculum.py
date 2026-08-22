"""Audit mina_6_8b JSON curriculum. Does not construct 6.8B."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from simulations.synthetic_world.curriculum_6_8b import PHASE_LENGTHS
from simulations.synthetic_world.dataset_v1 import REQUIRED_6_8B_KEYS, validate_episode_record

V03_MIN_EPISODES = 1000
V03_MIN_CORRECTIONS = 2000


def audit_curriculum(root: Path, *, gate: bool = False) -> dict:
    records = []
    for path in sorted(root.rglob("*.json")):
        if path.name in {"dataset_report.json"}:
            continue
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
    pwm = any(bool((r.get("embodiment") or {}).get("pwm")) for r in records)
    typed = Counter()
    for rec in records:
        for row in rec.get("corrections") or []:
            typed[str(row.get("correction_type") or row.get("lesson") or "untagged")] += 1
    report = {
        "dataset_root": str(root),
        "n_episodes": len(records),
        "phase_counts": dict(phase_counts),
        "scenario_counts": dict(scenario_counts),
        "transition_lengths": lengths,
        "observation_lengths": obs_lengths,
        "correction_count": correction_count,
        "correction_types": dict(typed),
        "event_count": sum(len(r.get("events", [])) for r in records),
        "future_diversity_mean": sum(diversities) / max(len(diversities), 1),
        "future_diversity_positive": sum(1 for d in diversities if d > 1e-6),
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
        },
    }
    if gate:
        failed = [name for name, ok in report["gate"].items() if not ok]
        if failed:
            raise SystemExit(f"v0.3 curriculum gate failed: {failed}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit MINA 6.8B JSON curriculum")
    parser.add_argument("--root", type=Path, default=Path("dataset/mina_6_8b_v03"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--gate", action="store_true", help="fail if v0.3 production thresholds miss")
    args = parser.parse_args()
    report = audit_curriculum(args.root, gate=args.gate)
    text = json.dumps(report, indent=2, sort_keys=True)
    out = args.out or (args.root / "dataset_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
