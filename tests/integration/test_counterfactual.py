"""Counterfactual: different actions produce different predicted futures."""

from __future__ import annotations

from minakanushi.future.engine import group_by_strategy
from minakanushi.strategy.candidate import StrategyCandidate
from tests.conftest import build_engine
from simulations.synthetic_world.world import SyntheticWorld


def test_different_actions_split_futures() -> None:
    engine = build_engine()
    world = SyntheticWorld(engine.config.simulation, seed=5)
    state = engine.initialize()
    obs = world.observe()
    result = engine.step(obs, state)
    a = StrategyCandidate("move_a", "MOVE_TO", (8.0, 2.0), 0.0, 0.0)
    b = StrategyCandidate("move_b", "MOVE_TO", (2.0, 8.0), 0.0, 0.0)
    futures = engine.future.predict(result.state.world, [a, b], max_horizon=6)
    grouped = group_by_strategy(futures)
    assert len(grouped["move_a"]) == engine.config.architecture.future_branches
    assert len(grouped["move_b"]) == engine.config.architecture.future_branches
    agent_a = grouped["move_a"][0].terminal_xy[0]
    agent_b = grouped["move_b"][0].terminal_xy[0]
    dist = float(((agent_a - agent_b).pow(2).sum()).sqrt().item())
    assert dist > 0.5
