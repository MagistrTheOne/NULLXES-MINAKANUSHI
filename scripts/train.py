"""Train MINAKANUSHI from a training YAML. Do not run unless authorized."""

from __future__ import annotations

import argparse
from pathlib import Path

from minakanushi.training.trainer import trainer_from_files


def main() -> None:
    parser = argparse.ArgumentParser(description="NULLXES MINAKANUSHI trainer")
    parser.add_argument("--config", default="configs/training/stage0_validation.yaml")
    parser.add_argument("--out", default="experiments/stage0")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    trainer = trainer_from_files(root, root / args.config)
    trainer.fit(root / args.out)


if __name__ == "__main__":
    main()
