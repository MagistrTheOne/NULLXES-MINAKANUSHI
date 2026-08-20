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

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "objective": self.objective,
            "target_state": [float(self.target_state[0]), float(self.target_state[1])],
            "parameters": dict(self.parameters),
            "confidence": float(self.confidence),
            "valid_until": float(self.valid_until),
            "abort_conditions": list(self.abort_conditions),
            "provenance": self.provenance,
            "extras": dict(self.extras),
        }

    @classmethod
    def from_dict(cls, raw: dict | None) -> ActionIntent | None:
        if not raw:
            return None
        tgt = raw.get("target_state", (0.0, 0.0))
        return cls(
            strategy_id=str(raw["strategy_id"]),
            objective=str(raw["objective"]),
            target_state=(float(tgt[0]), float(tgt[1])),
            parameters={str(k): float(v) for k, v in dict(raw.get("parameters") or {}).items()},
            confidence=float(raw.get("confidence", 0.0)),
            valid_until=float(raw.get("valid_until", 0.0)),
            abort_conditions=tuple(str(x) for x in raw.get("abort_conditions", ())),
            provenance=str(raw.get("provenance", "")),
            extras={str(k): str(v) for k, v in dict(raw.get("extras") or {}).items()},
        )
