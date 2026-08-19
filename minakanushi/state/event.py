"""First-class world events. Events are not entities."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorldEvent:
    event_id: int
    type: str
    start_time: float
    end_time: float | None
    participants: tuple[int, ...]
    location: tuple[float, float] | None
    confidence: float
    causal_parents: tuple[int, ...] = ()
    predicted_consequences: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)
