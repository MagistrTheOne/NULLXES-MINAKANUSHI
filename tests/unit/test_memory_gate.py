"""S4: learned memory gate + config-driven fusion gains."""

from __future__ import annotations

import dataclasses

import torch

from minakanushi.core.dynamic_world_core import DynamicWorldCore
from helpers import cpu_config


def test_mem_gate_bounds_and_norm() -> None:
    cfg = cpu_config().architecture
    dwc = DynamicWorldCore(cfg)
    latent = torch.randn(1, 4, cfg.latent_dim)
    mem = torch.randn(1, 4, cfg.latent_dim)
    gate = torch.sigmoid(dwc.mem_gate(torch.cat([latent, mem], dim=-1))).detach()
    assert float(gate.min()) >= 0.0 and float(gate.max()) <= 1.0
    fused = dwc.mem_norm(latent + gate * mem).detach()
    # LayerNorm output has ~zero mean / unit variance per slot.
    assert abs(float(fused.mean())) < 0.5
    assert float(fused.var(unbiased=False)) > 0.5


def test_fusion_defaults_reproduce_legacy() -> None:
    cfg = cpu_config().architecture
    assert cfg.memory_fusion.latent_keep == 0.7
    assert cfg.memory_fusion.working_gain == 0.2
    assert cfg.memory_fusion.xy_gain == 0.1


def test_latent_keep_one_ignores_memory() -> None:
    from minakanushi.architecture.mina_unit import MinaUnit, pack_units
    from minakanushi.state.constructor import StateConstructor, empty_world_state

    base = cpu_config().architecture
    cfg = dataclasses.replace(base, memory_fusion=dataclasses.replace(base.memory_fusion, latent_keep=1.0))
    ctor = StateConstructor(cfg)
    world = empty_world_state(cfg, 1, device=torch.device("cpu"), dtype=torch.float32)
    unit = MinaUnit(
        source_type="vector", source_id=2, timestamp=0.0, sequence_index=1,
        spatial_frame="arena", spatial_position=(2.0, 2.0, 0.0), spatial_valid=True,
        semantic_embedding=torch.zeros(cfg.latent_dim),
        confidence=0.9, uncertainty=0.1, persistence=0.8,
        entity_reference=11, relation_reference=0, kind="mover",
        arrival_time=0.0, source_rate=10.0,
    )
    packed = pack_units([unit], batch_index=0, max_units=cfg.max_observations,
                        latent_dim=cfg.latent_dim, episode_position=0.0,
                        now=0.0, device=torch.device("cpu"), dtype=torch.float32)
    world = ctor.apply(packed, world, packed.semantic_embedding)
    hints = torch.ones_like(world.latent_state)
    world2 = ctor.apply(packed, world, packed.semantic_embedding, memory_hints=hints)
    slot = int((world2.entity_id == 11).nonzero(as_tuple=False)[0, 1].item())
    # latent_keep=1.0 keeps observed-slot latent on the evidence path.
    assert torch.equal(world2.latent_state[0, slot], world.latent_state[0, slot])
