"""Uncertainty rises under missing or noisy evidence."""

from __future__ import annotations

import torch

from minakanushi.architecture.config import load_architecture
from minakanushi.architecture.mina_unit import MinaUnit, pack_units
from minakanushi.state.constructor import StateConstructor, empty_world_state
from minakanushi.uncertainty.engine import UncertaintyEngine
from tests.conftest import ROOT


def test_occlusion_increases_uncertainty() -> None:
    config = load_architecture(ROOT / "configs" / "architecture" / "cpu_dev.yaml")
    ctor = StateConstructor(config)
    engine = UncertaintyEngine(config)
    device = torch.device("cpu")
    dtype = torch.float32
    dim = config.latent_dim
    world = empty_world_state(config, 1, device=device, dtype=dtype)
    unit = MinaUnit(
        source_type="vector",
        source_id=2,
        timestamp=0.0,
        sequence_index=0,
        spatial_frame="arena",
        spatial_position=(2.0, 2.0, 0.0),
        spatial_valid=True,
        semantic_embedding=torch.zeros(dim),
        confidence=0.95,
        uncertainty=0.05,
        persistence=1.0,
        entity_reference=11,
        relation_reference=0,
        kind="mover",
    )
    packed = pack_units(
        [unit],
        batch_index=0,
        max_units=config.max_observations,
        latent_dim=dim,
        episode_position=0.0,
        now=0.0,
        device=device,
        dtype=dtype,
    )
    world = ctor.apply(packed, world, packed.semantic_embedding)
    u0 = engine(world, packed)
    slot = (world.entity_id == 11).nonzero(as_tuple=False)[0, 1]
    seen = float(u0.channels[0, slot, 0].item())
    empty = pack_units(
        [],
        batch_index=0,
        max_units=config.max_observations,
        latent_dim=dim,
        episode_position=1.0,
        now=0.1,
        device=device,
        dtype=dtype,
    )
    world = ctor.apply(empty, world, empty.semantic_embedding)
    u1 = engine(world, empty)
    missing = float(u1.channels[0, slot, 0].item())
    assert missing > seen
