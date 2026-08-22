"""hidden_correction levels: disappear / velocity / intent. Gate 03 length<=12 stays L1."""

from __future__ import annotations

from pathlib import Path

from minakanushi.architecture.config import load_simulation
from simulations.synthetic_world.dataset import generate_episode, hidden_level, training_frame

ROOT = Path(__file__).resolve().parents[2]


def _mover_vel(episode, frame: int):
    truth = episode.truth[frame]
    for i, kind in enumerate(truth.kind):
        if kind == "mover":
            return truth.vel[i]
    raise AssertionError("no mover")


def _mover_xy(episode, frame: int):
    truth = episode.truth[frame]
    for i, kind in enumerate(truth.kind):
        if kind == "mover":
            return truth.xy[i]
    raise AssertionError("no mover")


def test_hidden_levels_are_different_causes() -> None:
    assert hidden_level("hidden_correction") == 1
    assert hidden_level("hidden_correction_l2") == 2
    assert hidden_level("hidden_correction_l3") == 3
    sim = load_simulation(ROOT / "configs" / "simulation" / "milestone1.yaml")
    l1 = generate_episode(sim, seed=7, episode_index=0, length=32, scenario="hidden_correction")
    l2 = generate_episode(sim, seed=7, episode_index=0, length=32, scenario="hidden_correction_l2")
    l3 = generate_episode(sim, seed=7, episode_index=0, length=32, scenario="hidden_correction_l3")
    rev = training_frame("hidden_correction", 32)
    assert training_frame("hidden_correction", 12) == 6
    assert float(_mover_vel(l1, rev).sum()) == 0.0
    assert float(_mover_vel(l2, rev).sum()) != 0.0
    assert tuple(_mover_xy(l3, rev).tolist()) != tuple(_mover_xy(l1, rev).tolist())
