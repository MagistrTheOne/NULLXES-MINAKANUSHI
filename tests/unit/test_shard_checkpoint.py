"""Sharded *.mina roundtrip on cpu_dev. Does not construct 6.8B."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from minakanushi.architecture.model import MinakanushiSystem
from minakanushi.training.checkpoint import load_mina, save_mina
from minakanushi.training.shard import merge_tensor_maps, split_tensor_map
from helpers import cpu_config


def test_split_merge_preserves_linear_weights() -> None:
    module = nn.Linear(16, 8)
    shards = split_tensor_map(module.state_dict(), max_bytes=32)
    assert len(shards) >= 1
    restored = merge_tensor_maps(shards)
    for key, tensor in module.state_dict().items():
        assert torch.equal(tensor, restored[key])


def test_sharded_mina_resume_and_validation_restore(tmp_path: Path) -> None:
    cfg = cpu_config()
    system = MinakanushiSystem(cfg.architecture)
    opt = torch.optim.AdamW(system.parameters(), lr=1e-3)
    dummy = torch.ones_like(next(system.parameters()))
    next(system.parameters()).data.add_(0.5)
    opt.zero_grad()
    next(system.parameters()).grad = dummy
    opt.step()
    before = {k: v.detach().clone() for k, v in system.state_dict().items()}
    path = tmp_path / "resume_step10.mina"
    save_mina(
        path,
        system,
        optimizer=opt,
        shard_max_bytes=256,
        extras={"step": 10, "dataset_cursor": 10, "dataset_name": "mina_6_8b_curriculum"},
    )
    fresh = MinakanushiSystem(cfg.architecture)
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    manifest, payload = load_mina(path, fresh, optimizer=fresh_opt, return_payload=True)
    assert manifest["sharded"] is True
    assert manifest["train"]["step"] == 10
    assert manifest["train"]["dataset_cursor"] == 10
    assert payload["optimizer"] is not None
    for key, tensor in before.items():
        assert torch.equal(tensor, fresh.state_dict()[key])


def test_save_mina_non_rank0_does_not_write_or_dump_state_dict(tmp_path, monkeypatch) -> None:
    cfg = cpu_config()
    system = MinakanushiSystem(cfg.architecture)

    def boom(_self):
        raise AssertionError("non-rank0 must not call state_dict")

    monkeypatch.setattr("minakanushi.training.checkpoint.is_rank0", lambda: False)
    monkeypatch.setattr("minakanushi.training.checkpoint.dist_barrier", lambda: None)
    monkeypatch.setattr(type(system), "state_dict", boom)
    path = tmp_path / "rank1.mina"
    saved = save_mina(path, system, gathered=None)
    assert saved == path
    assert not path.exists()
