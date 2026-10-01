"""Value a strategy given situation and predicted trajectory."""

from __future__ import annotations

from minakanushi.future.trajectory import FutureTrajectory
from minakanushi.state.entity import AGENT_SLOT
from minakanushi.strategy.candidate import StrategyCandidate


def evaluate_value(
    candidate: StrategyCandidate,
    trajectory: FutureTrajectory | None,
    goal_xy: tuple[float, float],
    *,
    w_risk: float = 0.5,
    w_unc: float = 0.25,
    w_speed: float = 1.0,
    max_speed: float | None = None,
) -> float:
    """-dist - w_risk*risk - w_unc*unc [- w_speed*overspeed].

    Defaults reproduce the legacy formula. max_speed adds an overspeed
    penalty on the declared `parameters["speed"]` so speeders lose to
    compliant strategies even before the kernel rejects them.
    """
    if trajectory is None:
        return candidate.expected_value
    terminal_agent = trajectory.terminal_xy[AGENT_SLOT]
    gx = float(goal_xy[0]) - float(terminal_agent[0].item())
    gy = float(goal_xy[1]) - float(terminal_agent[1].item())
    dist = (gx * gx + gy * gy) ** 0.5
    uncertainty = float(trajectory.uncertainty.detach().item())
    value = -dist - w_risk * candidate.predicted_risk - w_unc * uncertainty
    if max_speed is not None:
        try:
            speed = float(candidate.parameters.get("speed", 0.0))
        except (AttributeError, TypeError, ValueError):
            speed = 0.0
        value -= w_speed * max(0.0, speed - float(max_speed))
    return value
