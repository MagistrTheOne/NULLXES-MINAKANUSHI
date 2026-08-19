"""P(future) is not a function of inverse uncertainty."""

from __future__ import annotations

import torch

from minakanushi.future.engine import group_by_strategy
from minakanushi.strategy.candidate import StrategyCandidate
from tests.conftest import build_engine
from simulations.synthetic_world.world import SyntheticWorld


def test_probability_not_inverse_uncertainty() -> None:
    engine = build_engine()
    world = SyntheticWorld(engine.config.simulation, seed=8)
    state = engine.initialize()
    obs = world.observe()
    result = engine.step(obs, state)
    cand = StrategyCandidate("wait", "WAIT", (float(world.agent.xy[0]), float(world.agent.xy[1])), 0.0, 0.0)
    futures = engine.future.predict(result.state.world, [cand], max_horizon=3)
    grouped = group_by_strategy(futures)
    branches = grouped["wait"]
    probs = torch.stack([b.probability for b in branches])
    uncs = torch.stack([b.uncertainty for b in branches])
    assert abs(float(probs.sum()) - 1.0) < 1e-5
    assert engine.future.branch_logit_head.weight.data_ptr() != engine.future.branch_unc_head.weight.data_ptr()
    assert probs.shape == uncs.shape
