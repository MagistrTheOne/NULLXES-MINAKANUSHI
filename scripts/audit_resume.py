"""Audit that a *.mina is a continuation payload, not weights-only.

Does not construct 6.8B. Reads manifest + zip inventory only.

    python scripts/audit_resume.py --mina minakanushi_stage0_step128.mina
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minakanushi.training.baseline import inspect_mina


def audit_resume_payload(path: Path) -> dict:
    inventory = inspect_mina(path)
    keys = inventory["resume_keys"]
    failed = [name for name, ok in keys.items() if not ok]
    if inventory["step"] is None:
        failed.append("step")
    report = {
        "path": str(path),
        "sha256": inventory["sha256"],
        "step": inventory["step"],
        "dataset_cursor": inventory["dataset_cursor"],
        "scheduler": inventory["scheduler"],
        "resume_keys": keys,
        "pass": not failed,
        "failed": failed,
        "clone_risk": not keys["optimizer"],
    }
    if failed:
        raise SystemExit(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mina", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_resume_payload(args.mina), indent=2))


if __name__ == "__main__":
    main()
