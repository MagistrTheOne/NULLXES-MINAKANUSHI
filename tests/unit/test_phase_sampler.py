"""Warm / intelligence mix is data sampling, not a new layer."""

from __future__ import annotations

from pathlib import Path

from minakanushi.architecture.config import load_simulation
from minakanushi.training.episode_dataset import JsonEpisodeDataset
from minakanushi.training.phase_sampler import (
    INTELLIGENCE_MIX,
    WARM_MIX,
    PhaseCurriculumSampler,
    mix_for_mode,
    mode_for_job_step,
)
from simulations.synthetic_world.curriculum_6_8b import write_curriculum

ROOT = Path(__file__).resolve().parents[2]


def test_warm_then_intelligence_mixes() -> None:
    assert mix_for_mode("warm") == WARM_MIX
    assert mix_for_mode("intelligence") == INTELLIGENCE_MIX
    assert abs(sum(WARM_MIX.values()) - 1.0) < 1e-9
    assert abs(sum(INTELLIGENCE_MIX.values()) - 1.0) < 1e-9
    assert mode_for_job_step(1, warm_steps=16) == "warm"
    assert mode_for_job_step(16, warm_steps=16) == "warm"
    assert mode_for_job_step(17, warm_steps=16) == "intelligence"


def test_sampler_is_deterministic_and_hits_all_phases(tmp_path: Path) -> None:
    config = load_simulation(ROOT / "configs" / "simulation" / "milestone1.yaml")
    write_curriculum(tmp_path, config, seed=7, n_episodes=1, length=8)
    ds = JsonEpisodeDataset(tmp_path, seed=11)
    sampler = PhaseCurriculumSampler(ds.paths, ds.phases, seed=11)
    a = [sampler.choose(step, "warm") for step in range(1, 80)]
    b = [sampler.choose(step, "warm") for step in range(1, 80)]
    assert a == b
    phases = {ds.phases[i] for i in a}
    assert phases == {"physics", "agency", "causality", "embodiment"}
