"""Read-only v0.3.1 dataset gate for H200. Does not generate or repair.

    python scripts/verify_v031_dataset.py --root dataset/mina_6_8b_v03
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minakanushi.training.v031_dataset import DatasetContractError, PROFILES, verify_v031_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("dataset/mina_6_8b_v03"))
    parser.add_argument("--profile", choices=PROFILES, default="v031")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    try:
        report = verify_v031_dataset(args.root, profile=args.profile, expected_seed=args.seed)
    except DatasetContractError as exc:
        print(json.dumps({"pass": False, "failures": exc.failures, "mutated": False}, indent=2, sort_keys=True))
        raise SystemExit("FAIL DATASET CONTRACT") from exc
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
