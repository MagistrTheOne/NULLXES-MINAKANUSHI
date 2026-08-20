"""Train MINAKANUSHI from a training YAML. Do not run unless authorized."""

from __future__ import annotations

import argparse
from pathlib import Path

from minakanushi.architecture.freeze import assert_may_construct
from minakanushi.training.trainer import trainer_from_files


def main() -> None:
    parser = argparse.ArgumentParser(description="NULLXES MINAKANUSHI trainer")
    parser.add_argument("--config", default="configs/training/stage0_validation.yaml")
    parser.add_argument("--out", default="experiments/stage0")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    from minakanushi.architecture.config import load_training, load_architecture

    training = load_training(root / args.config)
    arch = load_architecture(root / training.architecture)
    gpu_name = ""
    import torch

    if str(training.device).startswith("cuda") and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    assert_may_construct(arch, device=training.device, gpu_name=gpu_name)
    trainer = trainer_from_files(root, root / args.config)
    trainer.fit(root / args.out)


if __name__ == "__main__":
    main()
