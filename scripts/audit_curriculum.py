"""Audit mina_6_8b JSON curriculum. Does not construct 6.8B."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from simulations.synthetic_world.dataset_v1 import REQUIRED_6_8B_KEYS, validate_episode_record


def audit_curriculum(root: Path) -> dict:
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
    return {
        "dataset_root": str(root),
        "n_episodes": len(records),
        "phase_counts": dict(phase_counts),
        "scenario_counts": dict(scenario_counts),
        "transition_lengths": sorted({len(r.get("transitions", [])) for r in records}),
        "correction_count": sum(len(r.get("corrections", [])) for r in records),
        "event_count": sum(len(r.get("events", [])) for r in records),
        "required_keys": list(REQUIRED_6_8B_KEYS),
        "pwm": False,
        "source_of_truth": "dataset/mina_6_8b",
        "hf_role": "adapter_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit MINA 6.8B JSON curriculum")
    parser.add_argument("--root", type=Path, default=Path("dataset/mina_6_8b"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = audit_curriculum(args.root)
    text = json.dumps(report, indent=2, sort_keys=True)
    out = args.out or (args.root / "dataset_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
