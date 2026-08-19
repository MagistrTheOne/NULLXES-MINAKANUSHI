"""ActionIntent — cognitive output. Not motor PWM."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ActionIntent:
    strategy_id: str
    objective: str
    target_state: tuple[float, float]
    parameters: dict[str, float]
    confidence: float
    valid_until: float
    abort_conditions: tuple[str, ...]
    provenance: str
    extras: dict[str, str] = field(default_factory=dict)
