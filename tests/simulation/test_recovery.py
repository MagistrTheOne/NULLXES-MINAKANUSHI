"""Recovery: later evidence overwrites an incorrect world hypothesis."""

from __future__ import annotations

import torch

from minakanushi.architecture.config import load_architecture
from minakanushi.architecture.mina_unit import MinaUnit, pack_units
from minakanushi.state.constructor import StateConstructor, empty_world_state
from tests.conftest import ROOT


def test_later_evidence_corrects_position() -> None:
    config = load_architecture(ROOT / "configs" / "architecture" / "cpu_dev.yaml")
    ctor = StateConstructor(config)
    device = torch.device("cpu")
    dtype = torch.float32
    dim = config.latent_dim
    world = empty_world_state(config, 1, device=device, dtype=dtype)

    def unit(x: float, t: float) -> MinaUnit:
        return MinaUnit(
            source_type="vector",
            source_id=2,
            timestamp=t,
            sequence_index=0,
            spatial_frame="arena",
            spatial_position=(x, 3.0, 0.0),
            spatial_valid=True,
            semantic_embedding=torch.zeros(dim),
            confidence=0.99,
            uncertainty=0.01,
            persistence=1.0,
            entity_reference=11,
            relation_reference=0,
            kind="mover",
        )

    first = pack_units([unit(1.0, 0.0)], batch_index=0, max_units=8, latent_dim=dim, episode_position=0.0, now=0.0, device=device, dtype=dtype)
    world = ctor.apply(first, world, first.semantic_embedding)
    slot = (world.entity_id == 11).nonzero(as_tuple=False)[0, 1]
    assert abs(float(world.entity_xy[0, slot, 0]) - 1.0) < 1e-5
    second = pack_units([unit(6.0, 0.2)], batch_index=0, max_units=8, latent_dim=dim, episode_position=1.0, now=0.2, device=device, dtype=dtype)
    world = ctor.apply(second, world, second.semantic_embedding)
    slot = (world.entity_id == 11).nonzero(as_tuple=False)[0, 1]
    assert abs(float(world.entity_xy[0, slot, 0]) - 6.0) < 1e-5
