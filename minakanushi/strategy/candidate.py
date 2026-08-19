"""Strategy is a desired high-level transition, not an actuator command."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StrategyCandidate:
    strategy_id: str
    objective: str
    target_xy: tuple[float, float]
    expected_value: float
    uncertainty: float
    required_resources: tuple[str, ...] = ()
    predicted_risk: float = 0.0
    constraint_status: str = "unevaluated"
    parameters: dict[str, float] = field(default_factory=dict)
