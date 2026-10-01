"""H4: single speed contract + plant boundary re-check."""

from __future__ import annotations

import pytest
import torch

from minakanushi.architecture.config import SimulationConfig
from minakanushi.constraints.kernel import MinakanushiConstraintKernel
from minakanushi.policy.intent import ActionIntent
from minakanushi.runtime.platform import SyntheticPlatform
from minakanushi.strategy.candidate import StrategyCandidate
from simulations.synthetic_world.world import SyntheticWorld
from helpers import cpu_config


def _intent(objective: str, target, speed: float = 1.0) -> ActionIntent:
    return ActionIntent(
        strategy_id="t", objective=objective, target_state=target,
        parameters={"speed": speed}, confidence=1.0, valid_until=99.0,
        abort_conditions=(), provenance="test",
    )


def test_overspeed_intent_rejected_at_boundary() -> None:
    sim = SimulationConfig()
    platform = SyntheticPlatform(SyntheticWorld(sim, seed=1))
    before = tuple(float(v) for v in platform.world.agent.xy)
    result = platform.execute(_intent("MOVE_TO", (5.0, 5.0), speed=sim.max_speed + 1.0))
    assert result.applied is False
    assert tuple(float(v) for v in platform.world.agent.xy) == before


def test_outside_arena_intent_rejected_at_boundary() -> None:
    sim = SimulationConfig()
    platform = SyntheticPlatform(SyntheticWorld(sim, seed=1))
    result = platform.execute(_intent("MOVE_TO", (999.0, 999.0)))
    assert result.applied is False


def test_legal_intent_still_applies() -> None:
    sim = SimulationConfig()
    platform = SyntheticPlatform(SyntheticWorld(sim, seed=1))
    result = platform.execute(_intent("MOVE_TO", (5.0, 5.0)))
    assert result.applied is True


def test_max_speed_checks_declared_and_trajectory() -> None:
    from minakanushi.constraints.rule import MaxSpeed
    from minakanushi.future.trajectory import FutureTrajectory

    sim = SimulationConfig()
    rule = MaxSpeed()
    cand = StrategyCandidate("m", "MOVE_TO", (5.0, 5.0), 0.0, 0.0, parameters={"speed": 99.0})
    ok, _ = rule.evaluate(cand, None, sim)
    assert ok is False
    slow = StrategyCandidate("m", "MOVE_TO", (5.0, 5.0), 0.0, 0.0, parameters={"speed": 0.5})
    ok, _ = rule.evaluate(slow, None, sim)
    assert ok is True
    teleport = FutureTrajectory(
        states_xy=torch.tensor([[[0.0, 0.0]], [[9.0, 9.0]]]),
        probability=1.0, uncertainty=0.1, causal_assumptions=(),
        terminal_xy=torch.tensor([0.0, 0.0]),
        action_id="m", strategy_id="m", branch_id=0, horizon_steps=2,
        branch_logit=torch.tensor(0.0),
    )
    ok, reason = rule.evaluate(slow, teleport, sim)
    assert ok is False
    assert "trajectory peak" in reason


def test_future_action_vector_respects_max_speed() -> None:
    from minakanushi.future.engine import FutureEngine

    cfg = cpu_config().architecture
    engine = FutureEngine(cfg)
    agent = torch.tensor([[1.0, 1.0]])
    cand = StrategyCandidate("m", "MOVE_TO", (5.0, 5.0), 0.0, 0.0, parameters={"speed": 5.0})
    vec = engine._action_vector(cand, agent, max_speed=1.2)
    assert float(vec[0, :2].norm().item()) == pytest.approx(1.2, abs=1e-5)
    vec2 = engine._action_vector(cand, agent)
    assert float(vec2[0, :2].norm().item()) == pytest.approx(5.0, abs=1e-5)
