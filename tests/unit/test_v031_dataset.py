"""v0.3.1 dataset pack: CPU creates, H200 only verifies."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from minakanushi.architecture.config import load_simulation
from minakanushi.training.v031_dataset import (
    READY_NAME,
    DatasetContractError,
    _length_failures,
    assert_v031_train_dataset,
    prepare_v031_dataset,
    verify_v031_dataset,
)
from simulations.synthetic_world.curriculum_6_8b import PHASE_LENGTHS, PHASE_ORDER, episode_record

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def cpu_dev_pack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("mina_6_8b_v03")
    prepare_v031_dataset(root, n=10, seed=11, profile="cpu_dev")
    return root


def _clone(src: Path, dest: Path) -> Path:
    shutil.copytree(src, dest)
    return dest


def test_v031_length_gate_uses_obs_minus_one_transitions() -> None:
    """Curriculum writes length observations and length-1 transitions. cpu_dev skips this."""
    config = load_simulation(ROOT / "configs" / "simulation" / "milestone1.yaml")
    records = []
    for phase in PHASE_ORDER:
        rec = episode_record(config, phase=phase, seed=11, episode_index=0, length=PHASE_LENGTHS[phase])
        want = PHASE_LENGTHS[phase]
        assert len(rec["observations"]) == want
        assert len(rec["transitions"]) == want - 1
        records.append(rec)
    assert _length_failures(records, profile="v031") == []
    assert _length_failures(records, profile="cpu_dev") == []
    equal_counts = [
        dict(records[0], transitions=list(records[0]["observations"]))
    ]
    failed = _length_failures(equal_counts, profile="v031")
    assert failed
    assert "transitions" in failed[0]


def test_v031_profile_refuses_small_n(tmp_path: Path) -> None:
    with pytest.raises(DatasetContractError) as exc:
        prepare_v031_dataset(tmp_path / "pack", n=2, seed=11, profile="v031")
    assert any("n=2" in item for item in exc.value.failures)
    assert not (tmp_path / "pack" / READY_NAME).exists()


def test_cpu_dev_prepare_split_ready_and_verify_readonly(cpu_dev_pack: Path, tmp_path: Path) -> None:
    root = cpu_dev_pack
    assert (root / READY_NAME).is_file()
    assert (root / "dataset_manifest.json").is_file()
    assert (root / "dataset_report.json").is_file()
    for name in ("physics", "agency", "causality", "embodiment"):
        assert (root / name).is_dir()
    for rel in ("index.jsonl", "train/index.jsonl", "heldout/index.jsonl", "splits.json"):
        assert (root / rel).is_file()
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["pwm"] is False
    assert manifest["hf_data"] is False
    assert manifest["audit"] == "PASS"
    assert manifest["episodes"] == 40
    assert manifest["train"] == 36
    assert manifest["heldout"] == 4
    assert manifest["hashes"]["index.jsonl"]
    ready = json.loads((root / READY_NAME).read_text(encoding="utf-8"))
    assert ready["profile"] == "cpu_dev"
    assert ready["manifest_sha256"]
    ok = verify_v031_dataset(root, profile="cpu_dev", expected_seed=11)
    assert ok["mutated"] is False
    clone = _clone(root, tmp_path / "readonly")
    (clone / "heldout" / "index.jsonl").unlink()
    with pytest.raises(DatasetContractError):
        verify_v031_dataset(clone, profile="cpu_dev")
    assert not (clone / "heldout" / "index.jsonl").is_file()
    assert (root / "heldout" / "index.jsonl").is_file()


def test_cpu_dev_ready_does_not_unlock_h200_train(cpu_dev_pack: Path) -> None:
    training = SimpleNamespace(name="mina_6_8b_v03", dataset_root=str(cpu_dev_pack))
    with pytest.raises(DatasetContractError):
        assert_v031_train_dataset(ROOT, training)
    with pytest.raises(DatasetContractError):
        verify_v031_dataset(cpu_dev_pack, profile="v031")


def test_missing_ready_refuses_train(tmp_path: Path) -> None:
    training = SimpleNamespace(name="mina_6_8b_v03", dataset_root=str(tmp_path / "mina_6_8b_v03"))
    with pytest.raises(DatasetContractError) as exc:
        assert_v031_train_dataset(ROOT, training)
    assert any("READY" in item for item in exc.value.failures)
    other = SimpleNamespace(name="stage0_overfit", dataset_root=str(tmp_path))
    assert assert_v031_train_dataset(ROOT, other)["required"] is False


def test_pwm_in_action_fails_verify(cpu_dev_pack: Path, tmp_path: Path) -> None:
    root = _clone(cpu_dev_pack, tmp_path / "pwm")
    episode = next((root / "physics").glob("*.json"))
    rec = json.loads(episode.read_text(encoding="utf-8"))
    rec["actions"][0]["motor_left"] = 0.8
    episode.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(DatasetContractError) as exc:
        verify_v031_dataset(root, profile="cpu_dev")
    assert any("pwm" in item or "ActionIntent" in item for item in exc.value.failures)


def test_heldout_in_train_is_leak(cpu_dev_pack: Path, tmp_path: Path) -> None:
    root = _clone(cpu_dev_pack, tmp_path / "leak")
    held_line = (root / "heldout" / "index.jsonl").read_text(encoding="utf-8").splitlines()[0]
    train = root / "train" / "index.jsonl"
    train.write_text(train.read_text(encoding="utf-8") + held_line + "\n", encoding="utf-8")
    with pytest.raises(DatasetContractError) as exc:
        verify_v031_dataset(root, profile="cpu_dev")
    assert any("leak" in item for item in exc.value.failures)
