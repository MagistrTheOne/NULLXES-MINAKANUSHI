"""Train MINAKANUSHI from a training YAML. Do not run unless authorized."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from minakanushi.architecture.config import load_architecture, load_training
from minakanushi.architecture.freeze import assert_may_construct
from minakanushi.training.parallel import init_process_group_if_needed
from minakanushi.training.trainer import trainer_from_files
from minakanushi.training.v031_dataset import assert_v031_train_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="NULLXES MINAKANUSHI trainer")
    parser.add_argument("--config", default="configs/training/stage0_validation.yaml")
    parser.add_argument("--out", default="experiments/stage0")
    parser.add_argument("--resume", default=None, help="*.mina to continue (optimizer, RNG, cursor, identity)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    training = load_training(root / args.config)
    assert_v031_train_dataset(root, training)
    arch = load_architecture(root / training.architecture)
    init_process_group_if_needed(training.parallelism, training.device)
    gpu_name = ""
    if str(training.device).startswith("cuda") and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    assert_may_construct(arch, device=training.device, gpu_name=gpu_name)
    trainer = trainer_from_files(root, root / args.config)
    resume = Path(args.resume) if args.resume else None
    trainer.fit(root / args.out, resume=resume)


if __name__ == "__main__":
    main()
