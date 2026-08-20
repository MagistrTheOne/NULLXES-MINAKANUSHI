"""Frozen 6.8B profile. Load YAML only. Do not construct the net."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from minakanushi.architecture.config import load_architecture, load_training
from minakanushi.architecture.freeze import (
    FROZEN_AT,
    FROZEN_CORE_DEPTH,
    FROZEN_LATENT_DIM,
    FROZEN_PARAM_ESTIMATE,
    assert_6_8b_frozen,
    assert_may_construct,
    is_6_8b_profile,
)
from minakanushi.training.parallel import plan_from_training
from minakanushi.training.parameter_inventory import estimate_parameters

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_yaml_matches_7aba976_contract() -> None:
    cfg = load_architecture(ROOT / "configs" / "architecture" / "minakanushi_6_8b.yaml")
    pack = load_architecture(ROOT / "models" / "MINA-6.8B" / "architecture.yaml")
    assert_6_8b_frozen(cfg)
    assert_6_8b_frozen(pack)
    assert cfg.latent_dim == FROZEN_LATENT_DIM
    assert cfg.core_depth == FROZEN_CORE_DEPTH
    assert estimate_parameters(cfg)["total_estimate"] == FROZEN_PARAM_ESTIMATE
    assert FROZEN_AT == "7aba976"


def test_cpu_dev_is_not_the_6_8b_profile() -> None:
    cfg = load_architecture(ROOT / "configs" / "architecture" / "cpu_dev.yaml")
    assert not is_6_8b_profile(cfg)
    assert_may_construct(cfg, device="cpu")


def test_refuse_6_8b_construct_on_cpu() -> None:
    cfg = load_architecture(ROOT / "configs" / "architecture" / "minakanushi_6_8b.yaml")
    with pytest.raises(RuntimeError, match="CPU"):
        assert_may_construct(cfg, device="cpu")


def test_refuse_6_8b_on_blackwell_6000() -> None:
    cfg = load_architecture(ROOT / "configs" / "architecture" / "minakanushi_6_8b.yaml")
    with pytest.raises(RuntimeError, match="6000"):
        assert_may_construct(cfg, device="cuda", gpu_name="NVIDIA RTX PRO 6000 Blackwell Server Edition")


def test_refuse_single_h100_train() -> None:
    cfg = load_architecture(ROOT / "configs" / "architecture" / "minakanushi_6_8b.yaml")
    with pytest.raises(RuntimeError, match="H100"):
        assert_may_construct(cfg, device="cuda", gpu_name="NVIDIA H100", world_size=1)


def test_allow_h200() -> None:
    cfg = load_architecture(ROOT / "configs" / "architecture" / "minakanushi_6_8b.yaml")
    assert_may_construct(cfg, device="cuda", gpu_name="NVIDIA H200", world_size=2)


def test_sanity_yaml_is_fsdp2_bf16() -> None:
    arch = load_architecture(ROOT / "configs" / "architecture" / "minakanushi_6_8b.yaml")
    train = load_training(ROOT / "configs" / "training" / "mina_6_8b_sanity.yaml")
    plan = plan_from_training(arch, train)
    assert plan.parallelism == "fsdp2_zero3"
    assert plan.precision == "bf16"
    assert plan.activation_checkpoint is False
    assert plan.cognition_budget == 4
    assert train.checkpoint_every == train.steps


def test_fp16_is_forbidden_for_6_8b() -> None:
    arch = load_architecture(ROOT / "configs" / "architecture" / "minakanushi_6_8b.yaml")
    train = replace(load_training(ROOT / "configs" / "training" / "mina_6_8b_sanity.yaml"), precision="fp16")
    with pytest.raises(ValueError, match="FP16"):
        plan_from_training(arch, train)
