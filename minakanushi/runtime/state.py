"""RuntimeState — where the process is in its cycle. Not SelfModel. Not WorldState.

SelfModel: who I am.
WorldState: what is around.
RuntimeState: what this process is doing now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeState:
    cycle_id: int = 0
    runtime_time: float = 0.0
    mode: str = "AUTONOMOUS"
    current_situation_id: str = ""
    current_focus: dict[str, Any] = field(default_factory=dict)
    active_prediction: str = ""
    last_action_intent: dict[str, Any] | None = None
    last_action_result: dict[str, Any] | None = None
    pending_experience: int = 0
    health: str = "ok"
    checkpoint_reference: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": int(self.cycle_id),
            "runtime_time": float(self.runtime_time),
            "mode": str(self.mode),
            "current_situation_id": str(self.current_situation_id),
            "current_focus": dict(self.current_focus),
            "active_prediction": str(self.active_prediction),
            "last_action_intent": self.last_action_intent,
            "last_action_result": self.last_action_result,
            "pending_experience": int(self.pending_experience),
            "health": str(self.health),
            "checkpoint_reference": str(self.checkpoint_reference),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> RuntimeState:
        if not raw:
            return cls()
        return cls(
            cycle_id=int(raw.get("cycle_id", 0)),
            runtime_time=float(raw.get("runtime_time", 0.0)),
            mode=str(raw.get("mode", "AUTONOMOUS")),
            current_situation_id=str(raw.get("current_situation_id", "")),
            current_focus=dict(raw.get("current_focus") or {}),
            active_prediction=str(raw.get("active_prediction", "")),
            last_action_intent=raw.get("last_action_intent"),
            last_action_result=raw.get("last_action_result"),
            pending_experience=int(raw.get("pending_experience", 0)),
            health=str(raw.get("health", "ok")),
            checkpoint_reference=str(raw.get("checkpoint_reference", "")),
        )
