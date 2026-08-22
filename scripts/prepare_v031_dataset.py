"""Build the v0.3.1 dataset pack on CPU/2080. Does not construct 6.8B.

    python scripts/prepare_v031_dataset.py --root dataset/mina_6_8b_v03 --n 250 --seed 11
    python scripts/prepare_v031_dataset.py --root dataset/mina_6_8b_v03 --profile cpu_dev --n 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minakanushi.training.v031_dataset import DatasetContractError, PROFILES, prepare_v031_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("dataset/mina_6_8b_v03"))
    parser.add_argument("--n", type=int, default=250, help="episodes per phase. Production: 250. Not 2.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--profile", choices=PROFILES, default="v031")
    args = parser.parse_args()
    try:
        report = prepare_v031_dataset(args.root, n=args.n, seed=args.seed, profile=args.profile)
    except DatasetContractError as exc:
        print(json.dumps({"pass": False, "failures": exc.failures}, indent=2, sort_keys=True))
        raise SystemExit("FAIL DATASET CONTRACT") from exc
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    print("READY_FOR_H200" if args.profile == "v031" else "READY_CPU_DEV")


if __name__ == "__main__":
    main()
