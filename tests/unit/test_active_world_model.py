"""Gate 08 Active World Model: Belief(t)+Action → Belief(t+1)."""

from __future__ import annotations

import torch

from helpers import ROOT, build_engine
from minakanushi.architecture.config import load_architecture
from minakanushi.architecture.mina_unit import MinaUnit, pack_units
from minakanushi.future.engine import FutureEngine
from minakanushi.state.constructor import StateConstructor, empty_world_state
from minakanushi.strategy.candidate import StrategyCandidate
from minakanushi.training.metrics import action_influence_score, causal_consistency_score
from simulations.synthetic_world.world import SyntheticWorld


def _unit(config, x, y, vx, t, eid=11, conf=0.95):
    return MinaUnit(
        source_type="vector",
        source_id=2,
        timestamp=t,
        sequence_index=0,
        spatial_frame="arena",
        spatial_position=(x, y, 0.0),
        spatial_valid=True,
        semantic_embedding=torch.zeros(config.latent_dim),
        confidence=conf,
        uncertainty=1.0 - conf,
        persistence=1.0,
        entity_reference=eid,
        relation_reference=0,
        kind="mover",
        metadata={"vel": (vx, 0.0)},
    )


def _pack(config, units, now, ep, device, dtype):
    return pack_units(
        units,
        batch_index=0,
        max_units=config.max_observations,
        latent_dim=config.latent_dim,
        episode_position=ep,
        now=now,
        device=device,
        dtype=dtype,
    )


def _world_with_mover(config, vx: float):
    ctor = StateConstructor(config)
    device = torch.device("cpu")
    dtype = torch.float32
    packed = _pack(config, [_unit(config, 2.0, 1.0, vx, 0.0)], 0.0, 0.0, device, dtype)
    world = ctor.apply(packed, empty_world_state(config, 1, device=device, dtype=dtype), packed.semantic_embedding)
    return world, ctor, device, dtype


def test_08a_passive_coast_and_brake() -> None:
    config = load_architecture(ROOT / "configs" / "architecture" / "cpu_dev.yaml")
    moving, _, _, _ = _world_with_mover(config, 0.8)
    wait = StrategyCandidate("wait", "WAIT", (0.0, 0.0), 0.0, 0.0)
    future = FutureEngine(config)
    rolled = future.predict_belief(moving, wait, steps=4)
    slot = int((moving.entity_id == 11).nonzero(as_tuple=False)[0, 1].item())
    assert float(rolled.entity_xy[0, slot, 0]) > float(moving.entity_xy[0, slot, 0]) + 0.2
    assert float(rolled.existence[0, slot]) < float(moving.existence[0, slot])
    assert moving.entity_xy.data_ptr() != rolled.entity_xy.data_ptr()
    stopped, _, _, _ = _world_with_mover(config, 0.0)
    still = future.predict_belief(stopped, wait, steps=4)
    s2 = int((stopped.entity_id == 11).nonzero(as_tuple=False)[0, 1].item())
    assert abs(float(still.entity_xy[0, s2, 0]) - float(stopped.entity_xy[0, s2, 0])) < 0.05


def test_08b_wait_vs_move_split_agent_belief() -> None:
    config = load_architecture(ROOT / "configs" / "architecture" / "cpu_dev.yaml")
    world, _, _, _ = _world_with_mover(config, 0.2)
    future = FutureEngine(config)
    wait = StrategyCandidate("wait", "WAIT", (0.0, 0.0), 0.0, 0.0)
    move = StrategyCandidate("move", "MOVE_TO", (8.0, 1.0), 0.0, 0.0)
    a = future.predict_belief(world, wait, steps=6)
    b = future.predict_belief(world, move, steps=6)
    dist = float((a.entity_xy[0, 0] - b.entity_xy[0, 0]).pow(2).sum().sqrt())
    assert dist > 0.4
    infl = float(action_influence_score(a.entity_xy[0], b.entity_xy[0], a.occupied[0]))
    assert infl > 0.2


def test_08c_counterfactual_belief_branches() -> None:
    config = load_architecture(ROOT / "configs" / "architecture" / "cpu_dev.yaml")
    world, _, _, _ = _world_with_mover(config, 0.1)
    future = FutureEngine(config)
    a1 = StrategyCandidate("a1", "MOVE_TO", (8.0, 1.0), 0.0, 0.0)
    a2 = StrategyCandidate("a2", "MOVE_TO", (0.5, 8.0), 0.0, 0.0)
    b1 = future.predict_belief(world, a1, steps=5)
    b2 = future.predict_belief(world, a2, steps=5)
    assert not torch.allclose(b1.entity_xy[0, 0], b2.entity_xy[0, 0])


def test_08_causal_consistency_action_does_not_invent_mover_delta() -> None:
    config = load_architecture(ROOT / "configs" / "architecture" / "cpu_dev.yaml")
    world, _, _, _ = _world_with_mover(config, 0.3)
    future = FutureEngine(config)
    wait = StrategyCandidate("wait", "WAIT", (0.0, 0.0), 0.0, 0.0)
    move = StrategyCandidate("move", "MOVE_TO", (8.0, 1.0), 0.0, 0.0)
    a = future.predict_belief(world, wait, steps=5)
    b = future.predict_belief(world, move, steps=5)
    slot = int((world.entity_id == 11).nonzero(as_tuple=False)[0, 1].item())
    agent_delta = b.entity_xy[0, 0] - a.entity_xy[0, 0]
    mover_delta = b.entity_xy[0, slot] - a.entity_xy[0, slot]
    score = float(causal_consistency_score(agent_delta.unsqueeze(0), mover_delta.unsqueeze(0)))
    assert score > 0.2
    assert float(torch.linalg.vector_norm(mover_delta)) < 1e-5


def test_08d_action_outcome_after_closed_loop() -> None:
    engine = build_engine()
    sim = SyntheticWorld(engine.config.simulation, seed=12)
    state = engine.initialize()
    first = engine.step(sim.observe(), state)
    sim.step(first.action_intent)
    second = engine.step(sim.observe(), first.state)
    assert second.state.self_model is not None
    assert len(second.state.self_model.action_outcomes.records) >= 1
    rec = second.state.self_model.action_outcomes.records[-1]
    assert "objective" in rec.action_intent
    assert rec.prediction_error >= 0.0
    assert second.state.last_predicted is not None
    assert second.state.last_predicted.provenance == "future_belief"


def test_future_belief_does_not_write_world() -> None:
    engine = build_engine()
    sim = SyntheticWorld(engine.config.simulation, seed=3)
    state = engine.initialize()
    result = engine.step(sim.observe(), state)
    before = result.state.world.entity_xy.clone()
    ptr = result.state.world.entity_xy.data_ptr()
    move = StrategyCandidate("move", "MOVE_TO", (8.0, 2.0), 0.0, 0.0)
    _ = engine.future.predict_belief(result.state.world, move, steps=4)
    assert result.state.world.entity_xy.data_ptr() == ptr
    assert float((result.state.world.entity_xy - before).abs().sum()) == 0.0
