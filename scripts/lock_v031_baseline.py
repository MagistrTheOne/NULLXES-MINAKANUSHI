"""Lock the v0.3.1 pre-training origin. Never constructs 6.8B.

    python scripts/lock_v031_baseline.py \\
      --mina minakanushi_stage0_step128.mina \\
      --dataset dataset/mina_6_8b_v03 \\
      --out artifacts/v031/baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minakanushi.training.lock import lock_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mina", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/mina_6_8b_v03"))
    parser.add_argument("--config", type=Path, default=Path("configs/training/mina_6_8b_v03.yaml"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/v031/baseline"))
    parser.add_argument("--capability", type=Path, default=None)
    parser.add_argument("--run-capability", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--require-mina", action="store_true")
    args = parser.parse_args()
    report = lock_baseline(
        args.out,
        mina=args.mina,
        dataset_root=args.dataset,
        training_config=args.config,
        capability_path=args.capability,
        run_capability=bool(args.run_capability),
        write_inference=not args.skip_inference,
        require_mina=bool(args.require_mina),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report.get("checkpoint_sha256") == "MISSING" and args.require_mina:
        raise SystemExit("baseline lock missing step128 *.mina")


if __name__ == "__main__":
    main()
