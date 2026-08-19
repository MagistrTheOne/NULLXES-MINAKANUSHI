"""FutureEngine must not mutate WorldState tensors."""

from __future__ import annotations

from minakanushi.strategy.candidate import StrategyCandidate
from tests.conftest import build_engine
from simulations.synthetic_world.world import SyntheticWorld


def test_future_does_not_write_world_state() -> None:
    engine = build_engine()
    world = SyntheticWorld(engine.config.simulation, seed=4)
    state = engine.initialize()
    obs = world.observe()
    result = engine.step(obs, state)
    before_ptr = result.state.world.entity_xy.data_ptr()
    before_val = result.state.world.entity_xy.clone()
    cand = StrategyCandidate("move", "MOVE_TO", (8.0, 2.0), 0.0, 0.0)
    futures = engine.future.predict(result.state.world, [cand], max_horizon=4)
    futures[0].states_xy[0, 0, 0] = 99.0
    assert result.state.world.entity_xy.data_ptr() == before_ptr
    assert float((result.state.world.entity_xy - before_val).abs().sum()) == 0.0
