"""V2 future contract: scene evolution, not video prediction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SceneBranch:
    branch_id: str
    entities: tuple[str, ...]
    relations: tuple[str, ...]
    risk: float
    uncertainty: float
    suggested_hold: bool
    narrative: str

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.risk) <= 1.0:
            raise ValueError(f"risk out of range: {self.risk}")
        if not 0.0 <= float(self.uncertainty) <= 1.0:
            raise ValueError(f"uncertainty out of range: {self.uncertainty}")
        if "pixel" in self.narrative.lower() or "video" in self.narrative.lower():
            raise ValueError("SceneBranch is state prediction, not video generation")


def scene_branches(entities: tuple[str, ...], *, uncertainty: float) -> tuple[SceneBranch, ...]:
    names = tuple(entities) if entities else ("human", "door", "robot", "box")
    wait = SceneBranch(
        branch_id="A",
        entities=names,
        relations=("human_opens_door",),
        risk=0.2,
        uncertainty=min(1.0, max(0.0, float(uncertainty))),
        suggested_hold=False,
        narrative="human opens door → robot waits",
    )
    proceed = SceneBranch(
        branch_id="B",
        entities=names,
        relations=("human_walks_away",),
        risk=0.15,
        uncertainty=min(1.0, max(0.0, float(uncertainty))),
        suggested_hold=False,
        narrative="human walks away → robot proceeds",
    )
    hold = SceneBranch(
        branch_id="C",
        entities=names,
        relations=("uncertain",),
        risk=0.6,
        uncertainty=max(float(uncertainty), 0.7),
        suggested_hold=True,
        narrative="uncertainty high → SAFE_HOLD",
    )
    return (wait, proceed, hold)
