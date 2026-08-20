"""ActionPolicy cannot consume raw StrategyCandidate objects."""

from __future__ import annotations

import pytest

from minakanushi.future.engine import group_by_strategy
from minakanushi.strategy.candidate import StrategyCandidate
from helpers import build_engine
from simulations.synthetic_world.world import SyntheticWorld


def test_policy_rejects_raw_candidates() -> None:
    engine = build_engine()
    raw = StrategyCandidate("raid_restricted", "MOVE_TO", (8.2, 8.2), 100.0, 0.0)
    with pytest.raises(TypeError):
        engine.policy.select([raw], {}, (1.0, 1.0), 0.0)


def test_rejected_hard_strategy_cannot_be_selected() -> None:
    engine = build_engine()
    world = SyntheticWorld(engine.config.simulation, seed=1)
    state = engine.initialize()
    obs = world.observe()
    result = engine.step(obs, state)
    forbidden = StrategyCandidate("raid_restricted", "MOVE_TO", (8.2, 8.2), 100.0, 0.0)
    safe = StrategyCandidate("hold", "SAFE_HOLD", (float(world.agent.xy[0]), float(world.agent.xy[1])), -10.0, 0.0)
    futures = engine.future.predict(result.state.world, [forbidden, safe], max_horizon=8)
    allowed, rejected, audits = engine.constraints.filter([forbidden, safe], group_by_strategy(futures))
    assert forbidden.strategy_id in {c.strategy_id for c in rejected}
    intent = engine.policy.select(allowed, group_by_strategy(futures), engine.config.simulation.home, obs.timestamp)
    assert intent.strategy_id != "raid_restricted"
    assert any("restricted" in r for a in audits if not a.allowed for r in a.reasons)
