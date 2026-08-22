"""Freeze a *.mina baseline: sha256 + metrics_before.json.

Does not construct 6.8B. Reference inference is cpu_dev-only.

    python scripts/freeze_step128.py --mina minakanushi_stage0_step128.mina --out artifacts/v031/step128
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minakanushi.training.baseline import write_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mina", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/v031/step128"))
    args = parser.parse_args()
    report = write_baseline(args.mina, args.out)
    if report["research_scale"]:
        note = (
            "research-scale checkpoint: inference snapshot is load_mina on H200/B300, "
            "not constructed here. sha256 + manifest metrics are the laptop-safe anchor."
        )
        (args.out / "REFERENCE_INFERENCE.txt").write_text(note + "\n", encoding="utf-8")
        report["reference_inference_note"] = note
    print(json.dumps({k: report[k] for k in report if k != "members"}, indent=2, default=str))


if __name__ == "__main__":
    main()
