"""H3: FutureEngine non-mutation guard is value-based, UncertaintyEngine pure."""

from __future__ import annotations

import pytest
import torch

from minakanushi.architecture.config import load_architecture
from minakanushi.architecture.mina_unit import MinaUnit, pack_units
from minakanushi.future.engine import _assert_unmutated, _fingerprint
from minakanushi.state.constructor import StateConstructor, empty_world_state
from minakanushi.state.world import WorldState
from minakanushi.strategy.candidate import StrategyCandidate
from minakanushi.uncertainty.engine import UncertaintyEngine
from helpers import ROOT


def _world_with_mover():
    config = load_architecture(ROOT / "configs" / "architecture" / "cpu_dev.yaml")
    device = torch.device("cpu")
    dtype = torch.float32
    world = empty_world_state(config, 1, device=device, dtype=dtype)
    unit = MinaUnit(
        source_type="vector", source_id=2, timestamp=0.0, sequence_index=1,
        spatial_frame="arena", spatial_position=(2.0, 2.0, 0.0), spatial_valid=True,
        semantic_embedding=torch.zeros(config.latent_dim),
        confidence=0.9, uncertainty=0.1, persistence=0.8,
        entity_reference=11, relation_reference=0, kind="mover",
        arrival_time=0.0, source_rate=10.0,
    )
    packed = pack_units([unit], batch_index=0, max_units=config.max_observations,
                        latent_dim=config.latent_dim, episode_position=0.0,
                        now=0.0, device=device, dtype=dtype)
    world = StateConstructor(config).apply(packed, world, packed.semantic_embedding)
    return config, world, packed


def test_fingerprint_catches_inplace_mutation() -> None:
    _, world, _ = _world_with_mover()
    before = _fingerprint(world)
    _assert_unmutated(before, world, owner="probe")
    world.entity_xy.add_(1.0)
    with pytest.raises(RuntimeError, match="mutated WorldState"):
        _assert_unmutated(before, world, owner="probe")


def test_predict_and_predict_belief_do_not_mutate() -> None:
    from minakanushi.future.engine import FutureEngine

    config, world, _ = _world_with_mover()
    engine = FutureEngine(config)
    snap = {n: getattr(world, n).clone() for n in
            ("latent_state", "entity_xy", "entity_vel", "occupied", "uncertainty", "existence")}
    cand = StrategyCandidate("move_to_11", "MOVE_TO", (5.0, 5.0), 0.0, 0.0)
    engine.predict(world, [cand])
    engine.predict_belief(world, cand, steps=1)
    for name, value in snap.items():
        assert torch.equal(getattr(world, name), value), name


def test_uncertainty_forward_is_pure() -> None:
    config, world, packed = _world_with_mover()
    engine = UncertaintyEngine(config)
    before = world.uncertainty.clone()
    state = engine(world, packed)
    assert torch.equal(world.uncertainty, before)
    assert state.channels.shape[-1] == config.uncertainty_channels
