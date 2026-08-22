"""Write 6.8B pre-train episode curriculum. CPU. Does not construct 6.8B."""

from __future__ import annotations

import argparse
from pathlib import Path

from minakanushi.architecture.config import load_simulation
from simulations.synthetic_world.curriculum_6_8b import PHASE_LENGTHS, PHASE_ORDER, write_curriculum


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MINA 6.8B episode curriculum")
    parser.add_argument("--root", type=Path, default=Path("dataset/mina_6_8b_v03"))
    parser.add_argument(
        "--n",
        type=int,
        default=2,
        help="episodes per phase. Production pack: --n 250 (1000 total). JSON stays off git.",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=None,
        help="override all phases. Default v0.3: physics/agency 32, causality/embodiment 64.",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    config = load_simulation(repo / "configs" / "simulation" / "milestone1.yaml")
    written = write_curriculum(
        args.root,
        config,
        seed=args.seed,
        n_episodes=args.n,
        length=args.length,
        lengths=None if args.length is not None else PHASE_LENGTHS,
    )
    for phase in PHASE_ORDER:
        print(phase, len(written[phase]), "len", args.length or PHASE_LENGTHS[phase])


if __name__ == "__main__":
    main()
