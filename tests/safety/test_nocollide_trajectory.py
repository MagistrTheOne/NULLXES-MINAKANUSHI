"""NoCollideObstacle must reject trajectories through obstacles, not just targets.

v0.3.2 patch set. cpu_dev only. No 6.8B construct.
"""

from __future__ import annotations

import torch

from minakanushi.constraints.rule import NoCollideObstacle
from minakanushi.future.trajectory import FutureTrajectory
from minakanushi.state.entity import AGENT_SLOT
from minakanushi.strategy.candidate import StrategyCandidate
from helpers import cpu_config


def _rule():
    return NoCollideObstacle()


def _sim():
    return cpu_config().simulation


def _traj(points: list[tuple[float, float]], n_slots: int = 16) -> FutureTrajectory:
    h = len(points)
    states = torch.zeros(h, n_slots, 2)
    for t, (x, y) in enumerate(points):
        states[t, AGENT_SLOT, 0] = x
        states[t, AGENT_SLOT, 1] = y
    return FutureTrajectory(
        states_xy=states,
        probability=torch.tensor(1.0),
        uncertainty=torch.tensor(0.1),
        causal_assumptions=(),
        terminal_xy=states[-1].clone(),
        action_id="t",
        strategy_id="t",
        branch_id=0,
        horizon_steps=h,
    )


def test_target_inside_obstacle_rejected() -> None:
    sim = _sim()
    obs = sim.obstacles[0]
    ox, oy = float(obs["xy"][0]), float(obs["xy"][1])
    cand = StrategyCandidate("bad", "MOVE_TO", (ox, oy), 0.0, 0.0)
    ok, reason = _rule().evaluate(cand, None, sim)
    assert not ok
    assert "target inside obstacle" in reason


def test_trajectory_through_obstacle_rejected_despite_safe_target() -> None:
    sim = _sim()
    obs = sim.obstacles[0]
    ox, oy = float(obs["xy"][0]), float(obs["xy"][1])
    # Safe target far away, but straight-line rollout crosses the box.
    cand = StrategyCandidate("sneaky", "MOVE_TO", (1.0, 1.0), 0.0, 0.0)
    traj = _traj([(1.0, 1.0), (ox, oy), (1.0, 1.0)])
    ok, reason = _rule().evaluate(cand, traj, sim)
    assert not ok
    assert "trajectory step" in reason


def test_clear_trajectory_accepted() -> None:
    sim = _sim()
    cand = StrategyCandidate("clean", "MOVE_TO", (1.0, 1.0), 0.0, 0.0)
    traj = _traj([(1.0, 1.0), (1.1, 1.0), (1.0, 1.0)])
    ok, _ = _rule().evaluate(cand, traj, sim)
    assert ok
