"""Persistence: an entity leaving the observation set remains in WorldState."""

from __future__ import annotations

import torch

from minakanushi.architecture.config import load_architecture
from minakanushi.architecture.mina_unit import MinaUnit, pack_units
from minakanushi.state.constructor import StateConstructor, empty_world_state
from tests.conftest import ROOT


def _unit(eid: int, t: float, dim: int) -> MinaUnit:
    return MinaUnit(
        source_type="vector",
        source_id=2,
        timestamp=t,
        sequence_index=0,
        spatial_frame="arena",
        spatial_position=(4.0, 4.0, 0.0),
        spatial_valid=True,
        semantic_embedding=torch.ones(dim),
        confidence=0.9,
        uncertainty=0.1,
        persistence=1.0,
        entity_reference=eid,
        relation_reference=0,
        kind="mover",
    )


def test_entity_survives_temporary_observation_loss() -> None:
    config = load_architecture(ROOT / "configs" / "architecture" / "cpu_dev.yaml")
    ctor = StateConstructor(config)
    device = torch.device("cpu")
    dtype = torch.float32
    world = empty_world_state(config, 1, device=device, dtype=dtype)
    seen = pack_units(
        [_unit(11, 0.0, config.latent_dim)],
        batch_index=0,
        max_units=config.max_observations,
        latent_dim=config.latent_dim,
        episode_position=0.0,
        now=0.0,
        device=device,
        dtype=dtype,
    )
    world = ctor.apply(seen, world, seen.semantic_embedding)
    assert bool(((world.entity_id == 11) & world.occupied).any())
    empty = pack_units(
        [],
        batch_index=0,
        max_units=config.max_observations,
        latent_dim=config.latent_dim,
        episode_position=1.0,
        now=0.1,
        device=device,
        dtype=dtype,
    )
    world = ctor.apply(empty, world, empty.semantic_embedding)
    assert bool(((world.entity_id == 11) & world.occupied).any())
    assert float(world.age_unobserved[world.entity_id == 11].max()) >= 1.0
