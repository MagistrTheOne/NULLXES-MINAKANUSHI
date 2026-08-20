"""Sensor vs memory: not a blind average."""

from __future__ import annotations

import torch

from helpers import ROOT
from minakanushi.architecture.config import load_architecture
from minakanushi.architecture.mina_unit import MinaUnit, pack_units
from minakanushi.state.constructor import StateConstructor, empty_world_state
from minakanushi.training.metrics import evidence_dominance


def test_conflict_does_not_average() -> None:
    config = load_architecture(ROOT / "configs" / "architecture" / "cpu_dev.yaml")
    ctor = StateConstructor(config)
    device = torch.device("cpu")
    dtype = torch.float32
    dim = config.latent_dim
    world = empty_world_state(config, 1, device=device, dtype=dtype)

    def unit(x: float, t: float, conf: float, arrival: float) -> MinaUnit:
        return MinaUnit(
            source_type="vector",
            source_id=2,
            timestamp=t,
            sequence_index=0,
            spatial_frame="arena",
            spatial_position=(x, 5.0, 0.0),
            spatial_valid=True,
            semantic_embedding=torch.zeros(dim),
            confidence=conf,
            uncertainty=1.0 - conf,
            persistence=1.0,
            entity_reference=11,
            relation_reference=0,
            kind="mover",
            arrival_time=arrival,
            metadata={"vel": (0.0, 0.0)},
        )

    first = pack_units(
        [unit(1.0, 0.0, 0.8, 0.0)],
        batch_index=0,
        max_units=8,
        latent_dim=dim,
        episode_position=0.0,
        now=0.0,
        device=device,
        dtype=dtype,
    )
    world = ctor.apply(first, world, first.semantic_embedding)
    empty = pack_units(
        [],
        batch_index=0,
        max_units=8,
        latent_dim=dim,
        episode_position=1.0,
        now=0.1,
        device=device,
        dtype=dtype,
    )
    for _ in range(5):
        world = ctor.apply(empty, world, empty.semantic_embedding)
    slot = (world.entity_id == 11).nonzero(as_tuple=False)[0, 1]
    stale = float(world.entity_xy[0, slot, 0])
    sharp = pack_units(
        [unit(4.0, 2.0, 0.95, 2.0)],
        batch_index=0,
        max_units=8,
        latent_dim=dim,
        episode_position=21.0,
        now=2.0,
        device=device,
        dtype=dtype,
    )
    world = ctor.apply(sharp, world, sharp.semantic_embedding)
    result = float(world.entity_xy[0, slot, 0])
    mid = 0.5 * (stale + 4.0)
    assert evidence_dominance(result, stale, 4.0) == 1.0
    assert abs(result - mid) > abs(result - 4.0)
    assert world.uncertainty[0, slot, 2] > 0.0
    assert world.corrections
