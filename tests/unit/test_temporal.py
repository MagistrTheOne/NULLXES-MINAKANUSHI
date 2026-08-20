"""Temporal order: A then B is not B then A in world latent."""

from __future__ import annotations

import torch

from minakanushi.architecture.config import load_architecture
from minakanushi.architecture.mina_unit import MinaUnit, pack_units
from minakanushi.architecture.model import MinakanushiSystem
from minakanushi.state.constructor import StateConstructor, empty_world_state
from helpers import ROOT


def _unit(eid: int, x: float, t: float, dim: int) -> MinaUnit:
    return MinaUnit(
        source_type="vector",
        source_id=2,
        timestamp=t,
        sequence_index=0,
        spatial_frame="arena",
        spatial_position=(x, 1.0, 0.0),
        spatial_valid=True,
        semantic_embedding=torch.full((dim,), float(eid)),
        confidence=1.0,
        uncertainty=0.0,
        persistence=1.0,
        entity_reference=eid,
        relation_reference=0,
        kind="mover",
    )


def _run(order: list[int], config, system) -> torch.Tensor:
    device = torch.device("cpu")
    dtype = torch.float32
    ctor = StateConstructor(config)
    world = empty_world_state(config, 1, device=device, dtype=dtype)
    mem = torch.zeros_like(world.latent_state)
    for t, eid in enumerate(order):
        packed = pack_units(
            [_unit(eid, float(eid), float(t), config.latent_dim)],
            batch_index=0,
            max_units=config.max_observations,
            latent_dim=config.latent_dim,
            episode_position=float(t),
            now=float(t),
            device=device,
            dtype=dtype,
        )
        pos = system.position_units(packed)
        fused = packed.semantic_embedding + pos.embedding
        world = ctor.apply(packed, world, fused, mem)
        _, core = system.observe_to_core(packed, world, mem)
        world = core.world_state
        mem = torch.zeros_like(world.latent_state)
    return world.latent_state.detach()


def test_order_a_then_b_differs_from_b_then_a() -> None:
    config = load_architecture(ROOT / "configs" / "architecture" / "cpu_dev.yaml")
    system = MinakanushiSystem(config)
    system.eval()
    with torch.no_grad():
        ab = _run([11, 12], config, system)
        ba = _run([12, 11], config, system)
    assert not torch.allclose(ab, ba, atol=1e-5)
