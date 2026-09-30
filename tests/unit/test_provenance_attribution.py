"""Provenance + slot-error attribution. Diagnostic only, no semantics.

v0.3.2 instrumentation patch. No 6.8B construct. No training.
"""

from __future__ import annotations

from pathlib import Path

import torch

from minakanushi.training.episode_dataset import dataset_row_provenance
from minakanushi.training.metrics import slot_error_attribution

ROOT = Path(__file__).resolve().parents[2]


def test_row_provenance_deterministic() -> None:
    paths = (Path("physics/a.json"), Path("agency/b.json"))
    phases = ("physics", "agency")
    first = dataset_row_provenance(paths, phases, 1)
    second = dataset_row_provenance(paths, phases, 1)
    assert first == second
    assert first["train_row"] == 1
    assert first["phase"] == "agency"
    assert Path(first["episode_path"]).name == "b.json"
    assert Path(first["episode_path"]).parent.name == "agency"


def test_row_provenance_out_of_range() -> None:
    import pytest

    with pytest.raises(IndexError):
        dataset_row_provenance((Path("a.json"),), ("physics",), 5)


def _tensors():
    # 4 slots, slot 2 is the worst by far.
    pred = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [5.0, 0.0], [0.5, 0.0]]])
    true = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.5, 0.0]]])
    aligned = torch.tensor([[True, True, True, False]])
    eid = torch.tensor([[1, 7, 9, 3]])
    kind = torch.tensor([[1, 2, 2, 3]])
    occupied = torch.tensor([[True, True, True, True]])
    age = torch.tensor([[0.0, 0.0, 3.0, 0.0]])
    return pred, true, aligned, eid, kind, occupied, age


def test_slot_attribution_finds_max_slot() -> None:
    pred, true, aligned, eid, kind, occupied, age = _tensors()
    out = slot_error_attribution(
        pred, true, aligned, entity_id=eid, kind=kind, occupied=occupied, age_unobserved=age
    )
    assert out["top_error_slot"] == 2
    assert out["top_error"]["eid"] == 9
    assert out["top_error"]["eid_available"] is True
    assert out["top_error"]["kind"] == "mover"
    assert out["top_error"]["slot_error"] == 5.0
    assert out["top_error"]["occupied"] is True
    assert out["top_error"]["age_unobserved"] == 3.0
    assert out["top_error"]["observed"] is False
    assert out["n_aligned"] == 3
    assert out["n_unobserved"] == 1
    assert [r["slot"] for r in out["top3"]] == [2, 0, 1]


def test_slot_attribution_missing_metadata_safe() -> None:
    pred, true, aligned, _, _, _, _ = _tensors()
    out = slot_error_attribution(pred, true, aligned)
    assert out["top_error_slot"] == 2
    assert out["top_error"]["eid"] is None
    assert out["top_error"]["eid_available"] is False
    assert out["top_error"]["kind"] is None
    assert out["n_unobserved"] == 0


def test_synthetic_provenance_wiring() -> None:
    from minakanushi.training.trainer import trainer_from_files

    trainer = trainer_from_files(ROOT, ROOT / "configs" / "training" / "stage0_recurrent_t8_cpu.yaml")
    ep = trainer._load_episode(3, scenario="const_velocity", episode_index=None, seed=7, length=8)
    assert ep.scenario == "const_velocity"
    prov = trainer._last_prov
    assert prov["source"] == "synthetic"
    assert prov["train_row"] is None
    assert prov["episode_path"] is None
    assert prov["phase"] is None
