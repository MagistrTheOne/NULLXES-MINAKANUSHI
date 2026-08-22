"""Held-out split is by trajectory identity, not a file shuffle."""

from __future__ import annotations

from pathlib import Path

from minakanushi.architecture.config import load_simulation
from minakanushi.training.episode_dataset import JsonEpisodeDataset
from minakanushi.training.heldout import identity_key, is_heldout_index, write_heldout_split
from simulations.synthetic_world.curriculum_6_8b import write_curriculum

ROOT = Path(__file__).resolve().parents[2]


def test_heldout_uses_episode_index_not_shuffle(tmp_path: Path) -> None:
    config = load_simulation(ROOT / "configs" / "simulation" / "milestone1.yaml")
    write_curriculum(tmp_path, config, seed=7, n_episodes=10, length=8)
    report = write_heldout_split(tmp_path)
    assert report["n_episodes"] == 40
    assert report["n_heldout"] == 4
    assert report["n_train"] == 36
    assert report["leak"] is False
    train = JsonEpisodeDataset(tmp_path, seed=7, split="train")
    held = JsonEpisodeDataset(tmp_path, seed=7, split="heldout")
    assert len(train) == 36
    assert len(held) == 4
    train_ids = {identity_key(7, s, int(p.stem.rsplit("-", 1)[-1])) for p, s in zip(train.paths, train.scenarios)}
    held_ids = {identity_key(7, s, int(p.stem.rsplit("-", 1)[-1])) for p, s in zip(held.paths, held.scenarios)}
    assert not (train_ids & held_ids)
    for path in held.paths:
        episode_index = int(path.stem.rsplit("-", 1)[-1])
        assert is_heldout_index(episode_index)
        assert episode_index == 9


def test_reshuffled_index_does_not_move_trajectories(tmp_path: Path) -> None:
    config = load_simulation(ROOT / "configs" / "simulation" / "milestone1.yaml")
    write_curriculum(tmp_path, config, seed=7, n_episodes=10, length=8)
    first = write_heldout_split(tmp_path)
    index = tmp_path / "index.jsonl"
    lines = index.read_text(encoding="utf-8").splitlines()
    index.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
    second = write_heldout_split(tmp_path)
    assert first["n_train"] == second["n_train"]
    assert first["n_heldout"] == second["n_heldout"]
    held_a = {line for line in (tmp_path / "heldout" / "index.jsonl").read_text(encoding="utf-8").splitlines() if line}
    write_heldout_split(tmp_path)
    held_b = {line for line in (tmp_path / "heldout" / "index.jsonl").read_text(encoding="utf-8").splitlines() if line}
    ids_a = {line.split("episode_id")[-1] for line in held_a}
    ids_b = {line.split("episode_id")[-1] for line in held_b}
    assert ids_a == ids_b
