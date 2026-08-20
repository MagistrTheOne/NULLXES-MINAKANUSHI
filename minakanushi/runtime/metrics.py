"""Runtime process counters. Not training loss. Not desire."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuntimeMetrics:
    runtime_cycles: int = 0
    belief_updates: int = 0
    memory_writes: int = 0
    focus_changes: int = 0
    prediction_updates: int = 0
    action_attempts: int = 0
    authority_blocks: int = 0
    checkpoint_restores: int = 0
    experience_records: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "runtime_cycles": self.runtime_cycles,
            "belief_updates": self.belief_updates,
            "memory_writes": self.memory_writes,
            "focus_changes": self.focus_changes,
            "prediction_updates": self.prediction_updates,
            "action_attempts": self.action_attempts,
            "authority_blocks": self.authority_blocks,
            "checkpoint_restores": self.checkpoint_restores,
            "experience_records": self.experience_records,
        }

    @classmethod
    def from_dict(cls, raw: dict | None) -> RuntimeMetrics:
        if not raw:
            return cls()
        return cls(**{k: int(raw.get(k, 0)) for k in cls.__dataclass_fields__})
