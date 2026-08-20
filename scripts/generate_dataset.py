"""Generate NULLXES SyntheticWorld Dataset v1 on CPU. Not a training run."""

from __future__ import annotations

import argparse
from pathlib import Path

from minakanushi.architecture.config import load_config
from simulations.synthetic_world.dataset_v1 import SPLITS, write_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("dataset"))
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(
        root / "configs" / "architecture" / "cpu_dev.yaml",
        runtime_path=root / "configs" / "runtime" / "cpu.yaml",
        simulation_path=root / "configs" / "simulation" / "milestone1.yaml",
    )
    args.root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_split(args.root, split, cfg.simulation, seed=args.seed, n_episodes=args.n, length=args.length)
        print(split, "ok")


if __name__ == "__main__":
    main()
