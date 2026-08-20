"""Value a strategy given situation and predicted trajectory."""

from __future__ import annotations

from minakanushi.future.trajectory import FutureTrajectory
from minakanushi.state.entity import AGENT_SLOT
from minakanushi.strategy.candidate import StrategyCandidate


def evaluate_value(candidate: StrategyCandidate, trajectory: FutureTrajectory | None, goal_xy: tuple[float, float]) -> float:
    if trajectory is None:
        return candidate.expected_value
    terminal_agent = trajectory.terminal_xy[AGENT_SLOT]
    gx = float(goal_xy[0]) - float(terminal_agent[0].item())
    gy = float(goal_xy[1]) - float(terminal_agent[1].item())
    dist = (gx * gx + gy * gy) ** 0.5
    uncertainty = float(trajectory.uncertainty.detach().item())
    return -dist - 0.5 * candidate.predicted_risk - 0.25 * uncertainty
