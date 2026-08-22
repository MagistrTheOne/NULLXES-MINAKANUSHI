"""v0.3.2 diagnostic. No train. No 6.8B construct.

    python scripts/diagnose_counterfactual_v031.py
    python scripts/diagnose_counterfactual_v031.py --dataset dataset/mina_6_8b_v03
    python scripts/diagnose_counterfactual_v031.py --verdict artifacts/v031/verdict/step1128.json

H200 heldout forensic (optional, after metric fork A):

    python scripts/gate_v031_h200_verdict.py --before step128.mina --after step1128.mina
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minakanushi.training.counterfactual_forensic import (
    V031_LAST_STEP,
    V031_RESUME_START,
    diagnose,
    write_report,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--verdict", type=Path, default=None)
    parser.add_argument("--slots", type=int, default=512)
    parser.add_argument("--first-step", type=int, default=V031_RESUME_START)
    parser.add_argument("--last-step", type=int, default=V031_LAST_STEP)
    parser.add_argument("--resume-start", type=int, default=V031_RESUME_START)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "v032" / "diagnostic.json")
    args = parser.parse_args()
    report = diagnose(
        dataset=args.dataset,
        verdict_rows=args.verdict,
        slots=args.slots,
        first_step=args.first_step,
        last_step=args.last_step,
        resume_start=args.resume_start,
    )
    write_report(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    fork = report["metric"]["fork"]
    print(f"fork {fork} — {report['metric']['next']}")
    if fork != "A":
        raise SystemExit(f"v0.3.2 diagnostic fork {fork}")


if __name__ == "__main__":
    main()
