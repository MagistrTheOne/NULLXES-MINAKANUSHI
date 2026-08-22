"""Audit mina_6_8b JSON curriculum. Does not construct 6.8B.

    python scripts/audit_curriculum.py --root dataset/mina_6_8b_v03 --gate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minakanushi.training.curriculum_audit import audit_curriculum


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit MINA 6.8B JSON curriculum")
    parser.add_argument("--root", type=Path, default=Path("dataset/mina_6_8b_v03"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--split", type=str, default="", help="train | heldout | empty=all via root index")
    parser.add_argument("--gate", action="store_true", help="fail if v0.3 production thresholds miss")
    args = parser.parse_args()
    report = audit_curriculum(args.root, gate=args.gate, split=args.split)
    text = json.dumps(report, indent=2, sort_keys=True)
    out = args.out or (args.root / "dataset_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
