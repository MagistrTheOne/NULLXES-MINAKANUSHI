"""v0.3.1 baseline lock never constructs 6.8B."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from minakanushi.architecture.config import load_simulation, load_training
from minakanushi.training.lock import lock_baseline
from simulations.synthetic_world.curriculum_6_8b import write_curriculum

ROOT = Path(__file__).resolve().parents[2]


def test_lock_writes_origin_pack_without_mina(tmp_path: Path) -> None:
    config = load_simulation(ROOT / "configs" / "simulation" / "milestone1.yaml")
    data = tmp_path / "pack"
    write_curriculum(data, config, seed=7, n_episodes=1, length=8)
    out = tmp_path / "baseline"
    report = lock_baseline(
        out,
        mina=None,
        dataset_root=data,
        training_config=ROOT / "configs" / "training" / "mina_6_8b_v03.yaml",
        write_inference=False,
        run_capability=False,
        repo=ROOT,
    )
    assert report["constructed_6_8b"] is False
    assert report["checkpoint_sha256"] == "MISSING"
    assert (out / "checkpoint.sha256").read_text(encoding="utf-8").strip() == "MISSING"
    assert (out / "metrics_before.json").is_file()
    assert (out / "capability_before.json").is_file()
    assert (out / "dataset_report.json").is_file()
    assert (out / "training_config.yaml").is_file()
    assert (out / "git_commit.txt").is_file()
    assert (out / "hardware.json").is_file()
    assert (out / "run_manifest.json").is_file()
    assert (out / "git_status.json").is_file()
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"] == "MINAKANUSHI-6.8B"
    assert manifest["rgb"] is False
    assert manifest["pwm"] is False
    assert manifest["steps"] == 1000
    status = json.loads((out / "git_status.json").read_text(encoding="utf-8"))
    assert "code_dirty" in status
    dataset = json.loads((out / "dataset_report.json").read_text(encoding="utf-8"))
    assert "future_diversity_min" in dataset
    assert "revision_distribution" in dataset
    assert "max_action_fraction" in dataset
    train = load_training(out / "training_config.yaml")
    assert train.steps == 1000
    assert train.dataset_split == "train"


def test_v03_yaml_is_phase1_stop() -> None:
    train = load_training(ROOT / "configs" / "training" / "mina_6_8b_v03.yaml")
    assert train.steps == 1000
    assert train.eval_every == 50
    assert train.checkpoint_every == 250
    assert train.dataset_split == "train"
    assert train.dataset_root.replace("\\", "/").endswith("dataset/mina_6_8b_v03")
    assert train.sampler_mode == "auto"


def test_lock_require_dataset_refuses_pack_without_ready(tmp_path: Path) -> None:
    from minakanushi.training.v031_dataset import DatasetContractError

    config = load_simulation(ROOT / "configs" / "simulation" / "milestone1.yaml")
    data = tmp_path / "pack"
    write_curriculum(data, config, seed=7, n_episodes=1, length=8)
    with pytest.raises(DatasetContractError):
        lock_baseline(
            tmp_path / "baseline",
            mina=None,
            dataset_root=data,
            training_config=ROOT / "configs" / "training" / "mina_6_8b_v03.yaml",
            write_inference=False,
            require_dataset=True,
            repo=ROOT,
        )
