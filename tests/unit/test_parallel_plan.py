"""FSDP2 wrap, torchrun init, LOCAL_RANK bind, full-state gather."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from minakanushi.architecture.config import load_architecture, load_training
from minakanushi.training.parallel import (
    collect_full_checkpoint,
    init_process_group_if_needed,
    is_fsdp_wrapped,
    plan_from_training,
    training_device,
    wrap_fsdp2,
)

ROOT = Path(__file__).resolve().parents[2]


def test_wrap_fsdp2_requires_distributed() -> None:
    with pytest.raises(RuntimeError, match="torchrun"):
        wrap_fsdp2(nn.Linear(4, 4))


def test_6_8b_plan_does_not_construct_the_net() -> None:
    arch = load_architecture(ROOT / "configs" / "architecture" / "minakanushi_6_8b.yaml")
    train = load_training(ROOT / "configs" / "training" / "mina_6_8b_sanity.yaml")
    plan = plan_from_training(arch, train)
    assert plan.sharding == "fully_shard"
    assert plan.reduce_dtype == "fp32"


def test_fsdp2_yaml_requires_torchrun_env(monkeypatch) -> None:
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="torchrun"):
        init_process_group_if_needed("fsdp2_zero3", "cuda")


def test_training_device_binds_local_rank(monkeypatch) -> None:
    bound: list[int] = []
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda idx: bound.append(int(idx)))
    device = training_device("cuda")
    assert str(device) == "cuda:1"
    assert bound == [1]


def test_train_script_inits_process_group_before_construct() -> None:
    text = (ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    assert "init_process_group_if_needed" in text
    assert text.index("init_process_group_if_needed") < text.index("trainer_from_files")
    trainer = (ROOT / "minakanushi" / "training" / "trainer.py").read_text(encoding="utf-8")
    assert "collect_full_checkpoint" in trainer
    assert "gathered=gathered" in trainer


def test_collect_full_checkpoint_plain_module_roundtrip() -> None:
    module = nn.Linear(4, 2)
    with torch.no_grad():
        module.weight.fill_(3.0)
    opt = torch.optim.AdamW(module.parameters(), lr=1e-3)
    payload = collect_full_checkpoint(module, opt)
    assert payload is not None
    assert torch.equal(payload["system"]["weight"], module.weight.detach().cpu())
    assert payload["optimizer"] is not None
    assert payload["gathered"] is False


def test_fsdp_collect_gathers_full_state_dict(monkeypatch) -> None:
    module = nn.Linear(3, 1)
    captured: dict[str, object] = {}

    def fake_wrapped(_module: nn.Module) -> bool:
        return True

    def fake_get_model(_module, options=None):
        captured["full"] = bool(options.full_state_dict)
        captured["cpu"] = bool(options.cpu_offload)
        return {"weight": torch.ones(3, 1), "bias": torch.zeros(1)}

    def fake_get_opt(_module, _opt, options=None):
        return {"state": {}, "param_groups": []}

    monkeypatch.setattr("minakanushi.training.parallel.is_fsdp_wrapped", fake_wrapped)
    monkeypatch.setattr(
        "torch.distributed.checkpoint.state_dict.StateDictOptions",
        lambda **kwargs: SimpleNamespace(**kwargs),
        raising=False,
    )
    import torch.distributed.checkpoint.state_dict as sdict

    monkeypatch.setattr(sdict, "get_model_state_dict", fake_get_model)
    monkeypatch.setattr(sdict, "get_optimizer_state_dict", fake_get_opt)
    payload = collect_full_checkpoint(module, torch.optim.AdamW(module.parameters(), lr=1e-3))
    assert payload is not None
    assert captured["full"] is True
    assert captured["cpu"] is True
    assert payload["gathered"] is True
    assert payload["system"]["weight"].numel() == 3


def test_is_fsdp_wrapped_false_for_plain_linear() -> None:
    assert is_fsdp_wrapped(nn.Linear(2, 2)) is False


def test_wrap_fsdp2_shards_cognitive_blocks_not_root(monkeypatch) -> None:
    from minakanushi.architecture.model import MinakanushiSystem
    from minakanushi.core.cognitive_block import CognitiveBlock

    sharded: list[nn.Module] = []

    class _FakeMP:
        def __init__(self, **_kwargs) -> None:
            return

    def fake_fully_shard(mod, mp_policy=None):
        sharded.append(mod)
        return mod

    monkeypatch.setattr("torch.distributed.is_available", lambda: True)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "MixedPrecisionPolicy", _FakeMP, raising=False)
    monkeypatch.setattr(fsdp, "fully_shard", fake_fully_shard, raising=False)

    arch = load_architecture(ROOT / "configs" / "architecture" / "cpu_dev.yaml")
    system = MinakanushiSystem(arch)
    wrapped = wrap_fsdp2(system)
    assert wrapped is system
    assert sharded
    assert all(isinstance(mod, CognitiveBlock) for mod in sharded)
    assert system not in sharded
    assert system.perception not in sharded
