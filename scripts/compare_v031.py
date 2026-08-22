"""After H200: retention (old abilities) vs held-out (unseen episodes).

Not train-loss. Does not construct 6.8B.

    python scripts/compare_v031.py \\
      --before artifacts/v031/baseline/capability_before.json \\
      --after artifacts/v031/after/capability_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minakanushi.training.capability import compare_heldout, compare_retention, compare_ability_table


def _gates(payload: dict) -> dict:
    return payload.get("gates") or payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--verdict", type=Path, default=None, help="H200 artifacts/v031/verdict/compare.json")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    before = json.loads(args.before.read_text(encoding="utf-8"))
    after = json.loads(args.after.read_text(encoding="utf-8"))
    bg = _gates(before)
    ag = _gates(after)
    report = {
        "retention": compare_retention(bg["A"], ag["A"]) if "A" in bg and "A" in ag else {"status": "missing_gate_A"},
        "heldout": compare_heldout(bg["B"], ag["B"]) if "B" in bg and "B" in ag else {"status": "missing_gate_B"},
        "abilities": compare_ability_table(before, after),
        "not_a_claim": "train metrics up is not held-out up. hidden correction up with physics forgotten is not progress.",
        "do_not_compare": "step128 loss vs stepN loss",
    }
    if args.verdict is not None:
        h200 = json.loads(args.verdict.read_text(encoding="utf-8"))
        report["h200_heldout100"] = h200
        report["variant"] = h200.get("variant")
        report["accepted"] = bool(h200.get("accepted"))
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.verdict is not None:
        if not report.get("accepted"):
            raise SystemExit(f"v0.3.1 H200 verdict {report.get('variant')}")
        return
    retention_pass = report["retention"].get("pass", False)
    heldout_pass = report["heldout"].get("pass", False)
    if not (retention_pass and heldout_pass):
        raise SystemExit("v0.3.1 compare failed (retention or held-out)")


if __name__ == "__main__":
    main()
