"""Prediction: a moving entity produces a future trajectory."""

from __future__ import annotations

from minakanushi.strategy.candidate import StrategyCandidate
from tests.conftest import build_engine
from simulations.synthetic_world.world import SyntheticWorld


def test_moving_entity_yields_future_trajectory() -> None:
    engine = build_engine()
    world = SyntheticWorld(engine.config.simulation, seed=3)
    state = engine.initialize()
    for _ in range(6):
        obs = world.observe()
        result = engine.step(obs, state)
        state = result.state
        world.step(result.action_intent)
    cand = StrategyCandidate("wait", "WAIT", (float(world.agent.xy[0]), float(world.agent.xy[1])), 0.0, 0.0)
    futures = engine.future.predict(state.world, [cand], max_horizon=4)
    assert len(futures) == engine.config.architecture.future_branches
    assert futures[0].states_xy.shape[0] == 4
    mover_slots = (state.world.kind[0] == 2) & state.world.occupied[0]
    assert bool(mover_slots.any())
    start = futures[0].states_xy[0, mover_slots]
    end = futures[0].states_xy[-1, mover_slots]
    assert not (start - end).abs().sum() == 0
