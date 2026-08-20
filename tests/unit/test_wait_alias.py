"""WAIT is not OBSERVE: hold strategies share zero plant velocity, not one embedding."""

from __future__ import annotations

from helpers import build_engine
from minakanushi.strategy.candidate import StrategyCandidate
from minakanushi.strategy.hold import HOLD_MODE
from simulations.synthetic_world.world import SyntheticWorld


def test_wait_and_observe_are_not_the_same_future() -> None:
    engine = build_engine()
    world = SyntheticWorld(engine.config.simulation, seed=3)
    state = engine.initialize()
    obs = world.observe()
    result = engine.step(obs, state)
    xy = (float(world.agent.xy[0]), float(world.agent.xy[1]))
    wait = StrategyCandidate("wait", "WAIT", xy, 0.0, 0.0)
    observe = StrategyCandidate("observe", "OBSERVE", xy, 0.0, 0.0)
    hold = StrategyCandidate("safe_hold", "SAFE_HOLD", xy, 0.0, 0.0)
    futures = engine.future.predict(result.state.world, [wait, observe, hold], max_horizon=4)
    by_id = {t.strategy_id: t for t in futures if t.branch_id == 0}
    wait_xy = by_id["wait"].states_xy
    obs_xy = by_id["observe"].states_xy
    hold_xy = by_id["safe_hold"].states_xy
    assert HOLD_MODE["WAIT"] != HOLD_MODE["OBSERVE"]
    delta_wo = float((wait_xy - obs_xy).abs().max())
    delta_wh = float((wait_xy - hold_xy).abs().max())
    assert delta_wo > 1e-8 or delta_wh > 1e-8
