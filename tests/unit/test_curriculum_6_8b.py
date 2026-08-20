"""6.8B episode curriculum: observation → action → transition → lesson."""

from __future__ import annotations

from pathlib import Path

from minakanushi.architecture.config import load_simulation
from simulations.synthetic_world.curriculum_6_8b import (
    PHASE_ORDER,
    episode_record,
    write_curriculum,
)
from simulations.synthetic_world.replay import records_identical

ROOT = Path(__file__).resolve().parents[2]


def test_each_phase_emits_transitions_not_tokens(tmp_path: Path) -> None:
    config = load_simulation(ROOT / "configs" / "simulation" / "milestone1.yaml")
    for phase in PHASE_ORDER:
        record = episode_record(config, phase=phase, seed=7, episode_index=0, length=8)
        assert record["phase"] == phase
        assert record["transitions"]
        step = record["transitions"][0]
        for key in ("observation_t", "belief_t", "action_t", "observation_t1", "lesson"):
            assert key in step
        assert record["embodiment"]["pwm"] is False
        assert "token" not in record


def test_curriculum_replay_identity(tmp_path: Path) -> None:
    config = load_simulation(ROOT / "configs" / "simulation" / "milestone1.yaml")
    a = episode_record(config, phase="causality", seed=7, episode_index=1, length=8)
    b = episode_record(config, phase="causality", seed=7, episode_index=1, length=8)
    assert records_identical(a, b)


def test_write_curriculum_creates_four_phases(tmp_path: Path) -> None:
    config = load_simulation(ROOT / "configs" / "simulation" / "milestone1.yaml")
    written = write_curriculum(tmp_path, config, seed=7, n_episodes=1, length=8)
    assert tuple(written) == PHASE_ORDER
    for phase, paths in written.items():
        assert len(paths) == 1
        assert paths[0].is_file()
