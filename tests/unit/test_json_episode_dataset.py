"""JsonEpisodeDataset streams mina_6_8b records into Episode objects."""

from __future__ import annotations

from pathlib import Path

from minakanushi.architecture.config import load_simulation
from minakanushi.training.episode_dataset import JsonEpisodeDataset
from simulations.synthetic_world.curriculum_6_8b import write_curriculum
from simulations.synthetic_world.dataset_v1 import record_to_episode, validate_episode_record

ROOT = Path(__file__).resolve().parents[2]


def test_json_episode_dataset_streams_and_roundtrips(tmp_path: Path) -> None:
    config = load_simulation(ROOT / "configs" / "simulation" / "milestone1.yaml")
    written = write_curriculum(tmp_path, config, seed=7, n_episodes=1, length=8)
    assert all(written[phase] for phase in written)
    ds = JsonEpisodeDataset(tmp_path, seed=7)
    assert len(ds) == 4
    first = ds.episode(0)
    rec = ds.record(0)
    validate_episode_record(rec, curriculum_6_8b=True)
    again = record_to_episode(rec, curriculum_6_8b=True)
    assert first.scenario == again.scenario
    assert first.observations[0].agent_xy == again.observations[0].agent_xy
    assert len(first.truth) == len(again.truth)
    ds2 = JsonEpisodeDataset(tmp_path, seed=7)
    assert ds.paths == ds2.paths


def test_trainer_unroll_consumes_json(tmp_path: Path) -> None:
    from dataclasses import replace

    from minakanushi.architecture.config import load_config
    from minakanushi.training.trainer import Trainer

    config = load_simulation(ROOT / "configs" / "simulation" / "milestone1.yaml")
    write_curriculum(tmp_path, config, seed=7, n_episodes=1, length=8)
    cfg = load_config(
        ROOT / "configs" / "architecture" / "cpu_dev.yaml",
        training_path=ROOT / "configs" / "training" / "stage0_overfit.yaml",
        runtime_path=ROOT / "configs" / "runtime" / "cpu.yaml",
        simulation_path=ROOT / "configs" / "simulation" / "milestone1.yaml",
    )
    cfg = replace(cfg, training=replace(cfg.training, dataset_root=str(tmp_path), sequence_length=8))
    trainer = Trainer(cfg, ROOT)
    assert trainer.dataset is not None
    pkt = trainer.unroll(1)
    json_episode = trainer.dataset.episode(0)
    assert pkt.scenario == json_episode.scenario
    assert pkt.episode_index == json_episode.episode_index
