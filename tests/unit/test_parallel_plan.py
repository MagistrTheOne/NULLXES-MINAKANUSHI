"""FSDP2 wrap refuses to run without a process group. Plan is CPU-testable."""

from __future__ import annotations

from pathlib import Path

import pytest
from torch import nn

from minakanushi.architecture.config import load_architecture, load_training
from minakanushi.training.parallel import plan_from_training, wrap_fsdp2

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
