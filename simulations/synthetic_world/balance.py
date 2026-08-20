"""Dataset balance — catch constant-velocity collapse in the teacher."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BalanceReport:
    scenario_count: dict[str, int]
    event_count: int
    occlusion_count: int
    action_count: dict[str, int]
    correction_count: int
    conflict_count: int
    n_episodes: int

    def max_scenario_fraction(self) -> float:
        total = sum(self.scenario_count.values()) or 1
        return max(self.scenario_count.values()) / total if self.scenario_count else 0.0

    def constant_velocity_collapsed(self, *, limit: float = 0.90) -> bool:
        total = self.n_episodes or 1
        return self.scenario_count.get("const_velocity", 0) / total >= limit


def tally_records(records: list[dict[str, Any]]) -> BalanceReport:
    scenarios: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    events = 0
    occlusions = 0
    corrections = 0
    conflicts = 0
    for rec in records:
        scenarios[str(rec.get("scenario", "unknown"))] += 1
        for act in rec.get("actions", []):
            actions[str(act.get("objective", "NONE"))] += 1
        for ev in rec.get("events", []):
            events += 1
            kind = str(ev.get("type", ""))
            if kind == "occlusion":
                occlusions += 1
            if kind == "conflict":
                conflicts += 1
        corrections += len(rec.get("corrections", []))
    return BalanceReport(
        scenario_count=dict(scenarios),
        event_count=events,
        occlusion_count=occlusions,
        action_count=dict(actions),
        correction_count=corrections,
        conflict_count=conflicts,
        n_episodes=len(records),
    )


def load_records(root: Path) -> list[dict[str, Any]]:
    import json

    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows
