"""Milestone-1 scenario factory."""

from __future__ import annotations

from pathlib import Path

from minakanushi.architecture.config import load_simulation
from simulations.synthetic_world.world import SyntheticWorld


def build_milestone1(seed: int = 7) -> SyntheticWorld:
    path = Path(__file__).resolve().parents[2] / "configs" / "simulation" / "milestone1.yaml"
    return SyntheticWorld(load_simulation(path), seed=seed)
