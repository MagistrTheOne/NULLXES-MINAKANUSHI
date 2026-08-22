"""Capability protocol A–G. cpu_dev. Does not construct 6.8B.

    python scripts/gate_capability.py --out artifacts/v031/capability
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from minakanushi.training.capability import run_capability_suite


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().float().mean().item())
    if isinstance(value, float):
        return value
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/v031/capability"))
    args = parser.parse_args()
    report = run_capability_suite(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    payload = _jsonable(report)
    (args.out / "capability_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    unproven = payload.get("unproven") or []
    if unproven:
        print(f"protocol complete; capabilities not proven: {unproven}")
        print("do not update ledger PASS. H200 may still measure the same gates.")
    if not report.get("protocol_complete"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
