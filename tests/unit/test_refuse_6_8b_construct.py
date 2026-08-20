"""Trainer must not construct 6.8B on CPU. Activation checkpoint is a wrap, not a new module."""

from __future__ import annotations

import pytest
import torch

from helpers import ROOT, cpu_config
from minakanushi.architecture.mina_unit import pack_units
from minakanushi.architecture.model import MinakanushiSystem
from minakanushi.perception.bridge import Observation
from minakanushi.state.constructor import empty_world_state
from minakanushi.training.trainer import trainer_from_files


def test_trainer_refuses_6_8b_sanity_yaml_on_cpu() -> None:
    with pytest.raises(RuntimeError, match="GPU name|6000|torchrun|CPU|H100"):
        trainer_from_files(ROOT, ROOT / "configs" / "training" / "mina_6_8b_sanity.yaml")


def test_activation_checkpoint_forward_backward_on_cpu_dev() -> None:
    cfg = cpu_config().architecture
    system = MinakanushiSystem(cfg)
    system.train()
    system.world_core.activation_checkpoint = True
    device = torch.device("cpu")
    dtype = torch.float32
    obs = Observation(
        timestamp=0.0,
        agent_xy=(1.0, 1.0),
        agent_vel=(0.0, 0.0),
        heading=0.0,
        health=1.0,
        battery=1.0,
        visible=({"id": 2, "kind": "mover", "xy": (2.0, 1.0), "vel": (0.1, 0.0)},),
        occluded_ids=(),
        noise_std=0.0,
        arrival_time=0.0,
        source_rate_telemetry=20.0,
        source_rate_vector=10.0,
    )
    units = system.perception.encode(obs, device=device, dtype=dtype)
    batch = pack_units(
        units,
        batch_index=0,
        max_units=cfg.max_observations,
        latent_dim=cfg.latent_dim,
        episode_position=0.0,
        now=0.0,
        device=device,
        dtype=dtype,
    )
    world = empty_world_state(cfg, 1, device=device, dtype=dtype)
    world.occupied[0, 0] = True
    world.entity_id[0, 0] = 1
    hints = torch.zeros_like(world.latent_state)
    _, core = system.observe_to_core(batch, world, hints)
    loss = core.world_state.latent_state.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in system.world_core.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_dwc_checkpoint_passes_amp_recompute_context() -> None:
    block = (ROOT / "minakanushi" / "core" / "cognitive_block.py").read_text(encoding="utf-8")
    dwc = (ROOT / "minakanushi" / "core" / "dynamic_world_core.py").read_text(encoding="utf-8")
    parallel = (ROOT / "minakanushi" / "training" / "parallel.py").read_text(encoding="utf-8")
    trainer = (ROOT / "minakanushi" / "training" / "trainer.py").read_text(encoding="utf-8")
    assert "checkpoint(" in block
    assert "_compute_with_amp" in block
    assert "def cognitive_blocks" in dwc
    assert "checkpoint_wrapper" in parallel
    assert "activation_checkpoint=bool(train.activation_checkpoint)" in trainer
    assert "or is_6_8b_profile(config.architecture)" not in trainer


def test_activation_checkpoint_propagates_to_blocks() -> None:
    cfg = cpu_config().architecture
    system = MinakanushiSystem(cfg)
    assert system.world_core.blocks[0].activation_checkpoint is False
    system.world_core.activation_checkpoint = True
    system.world_core.checkpoint_amp_dtype = torch.bfloat16
    assert all(block.activation_checkpoint for block in system.world_core.blocks)
    assert all(block.amp_dtype is torch.bfloat16 for block in system.world_core.blocks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 autocast checkpoint")
def test_checkpoint_backward_outside_autocast_keeps_bf16() -> None:
    cfg = cpu_config().architecture
    system = MinakanushiSystem(cfg).to("cuda")
    system.train()
    system.world_core.activation_checkpoint = True
    system.world_core.checkpoint_amp_dtype = torch.bfloat16
    device = torch.device("cuda")
    dtype = torch.float32
    obs = Observation(
        timestamp=0.0,
        agent_xy=(1.0, 1.0),
        agent_vel=(0.0, 0.0),
        heading=0.0,
        health=1.0,
        battery=1.0,
        visible=({"id": 2, "kind": "mover", "xy": (2.0, 1.0), "vel": (0.1, 0.0)},),
        occluded_ids=(),
        noise_std=0.0,
        arrival_time=0.0,
        source_rate_telemetry=20.0,
        source_rate_vector=10.0,
    )
    units = system.perception.encode(obs, device=device, dtype=dtype)
    batch = pack_units(
        units,
        batch_index=0,
        max_units=cfg.max_observations,
        latent_dim=cfg.latent_dim,
        episode_position=0.0,
        now=0.0,
        device=device,
        dtype=dtype,
    )
    world = empty_world_state(cfg, 1, device=device, dtype=dtype)
    world.occupied[0, 0] = True
    world.entity_id[0, 0] = 1
    hints = torch.zeros_like(world.latent_state)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, core = system.observe_to_core(batch, world, hints)
    core.world_state.latent_state.float().pow(2).mean().backward()
    grads = [p.grad for p in system.world_core.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
