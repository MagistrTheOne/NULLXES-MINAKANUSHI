"""Focus Engine — internal attention selection. Not curiosity, not a goal generator.

Question answered: where is additional observation most valuable?
Not: what does MINA want.

Rule-based. No RL, no network, no drives.
Focus never writes ActionIntent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from minakanushi.state.correction import CONFLICT_CHANNEL, MISSING_CHANNEL
from minakanushi.state.entity import AGENT_SLOT
from minakanushi.state.world import WorldState

if TYPE_CHECKING:
    from minakanushi.identity.experience import ExperienceLog

MAINTENANCE_THRESHOLD = 0.30
FOCUS_TTL = 1.0
RESOURCE_COST = 0.05


class FocusType(str, Enum):
    UNCERTAINTY_REDUCTION = "UNCERTAINTY_REDUCTION"
    PREDICTION_ERROR = "PREDICTION_ERROR"
    NOVELTY = "NOVELTY"
    MEMORY_CONFLICT = "MEMORY_CONFLICT"
    MAINTENANCE = "MAINTENANCE"


@dataclass
class FocusState:
    """Attention state. Not a goal. Not an action."""

    target_id: int = 0
    focus_type: str = FocusType.MAINTENANCE.value
    priority: float = 0.0
    confidence: float = 1.0
    created_at: float = 0.0
    expires_at: float = 0.0

    @property
    def attention_target(self) -> str:
        if self.target_id == 0:
            return "none"
        return f"entity_{self.target_id}"

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "focus_type": self.focus_type,
            "priority": self.priority,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "attention_target": self.attention_target,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> FocusState:
        if not raw:
            return cls()
        if "target_id" not in raw and "attention_target" in raw:
            token = str(raw.get("attention_target", "none"))
            target = 0
            if token.startswith("entity_"):
                try:
                    target = int(token.split("_", 1)[1])
                except ValueError:
                    target = 0
            return cls(target_id=target, focus_type=FocusType.MAINTENANCE.value)
        return cls(
            target_id=int(raw.get("target_id", 0)),
            focus_type=str(raw.get("focus_type", FocusType.MAINTENANCE.value)),
            priority=float(raw.get("priority", 0.0)),
            confidence=float(raw.get("confidence", 1.0)),
            created_at=float(raw.get("created_at", 0.0)),
            expires_at=float(raw.get("expires_at", 0.0)),
        )


@dataclass
class FocusCandidate:
    entity_id: int
    focus_type: str
    uncertainty_gain: float
    prediction_error: float
    novelty: float
    mission_relevance: float
    risk: float
    resource_cost: float
    score: float


def score_focus(
    *,
    uncertainty_gain: float,
    prediction_error: float,
    novelty: float,
    mission_relevance: float,
    risk: float,
    resource_cost: float = RESOURCE_COST,
) -> float:
    return (
        uncertainty_gain
        + prediction_error
        + novelty
        + mission_relevance
        - risk
        - resource_cost
    )


class FocusEngine:
    """Generate candidates from belief, score them, emit one FocusState."""

    def generate(
        self,
        world: WorldState,
        log: ExperienceLog | None = None,
    ) -> list[FocusCandidate]:
        from minakanushi.identity.experience import ExperienceLog as ExperienceLogType

        log = log or ExperienceLogType()
        candidates: list[FocusCandidate] = []
        occ = world.occupied[0]
        for slot in occ.nonzero(as_tuple=False).flatten().tolist():
            if slot == AGENT_SLOT:
                continue
            eid = int(world.entity_id[0, slot].item())
            if eid == 0:
                continue
            observed = float(world.age_unobserved[0, slot].item()) == 0.0
            existence = float(world.existence[0, slot].item())
            xy_std = float(world.xy_std[0, slot].mean().item())
            missing = float(world.uncertainty[0, slot, MISSING_CHANNEL].item())
            conflict = float(world.uncertainty[0, slot, CONFLICT_CHANNEL].item())
            rec = log.latest_for(eid)
            err = 0.0
            if rec is not None:
                err = min(1.0, rec.error_xy + rec.error_vel)
            novelty = max(0.0, 1.0 - existence)
            if rec is None and existence < 0.75:
                novelty = max(novelty, 0.55)
            uncertainty_gain = min(1.0, 0.5 * missing + 0.5 * min(1.0, xy_std))
            if not observed:
                uncertainty_gain = min(1.0, uncertainty_gain + 0.25 * missing)
            mission_relevance = 0.0
            risk = 0.0
            memory_conflict = (not observed) and conflict >= 0.2
            if memory_conflict:
                kind = FocusType.MEMORY_CONFLICT.value
            elif err >= 0.25:
                kind = FocusType.PREDICTION_ERROR.value
            elif novelty >= 0.4:
                kind = FocusType.NOVELTY.value
            else:
                kind = FocusType.UNCERTAINTY_REDUCTION.value
            scored = score_focus(
                uncertainty_gain=uncertainty_gain,
                prediction_error=err,
                novelty=novelty,
                mission_relevance=mission_relevance,
                risk=risk,
            )
            if memory_conflict:
                scored = scored + conflict
            candidates.append(
                FocusCandidate(
                    entity_id=eid,
                    focus_type=kind,
                    uncertainty_gain=uncertainty_gain,
                    prediction_error=err,
                    novelty=novelty,
                    mission_relevance=mission_relevance,
                    risk=risk,
                    resource_cost=RESOURCE_COST,
                    score=scored,
                )
            )
        return candidates

    def select(
        self,
        world: WorldState,
        log: ExperienceLog | None = None,
        now: float = 0.0,
    ) -> FocusState:
        candidates = self.generate(world, log)
        if not candidates:
            return FocusState(
                target_id=0,
                focus_type=FocusType.MAINTENANCE.value,
                priority=0.0,
                confidence=1.0,
                created_at=now,
                expires_at=now + FOCUS_TTL,
            )
        best = max(candidates, key=lambda c: c.score)
        if best.score < MAINTENANCE_THRESHOLD:
            return FocusState(
                target_id=0,
                focus_type=FocusType.MAINTENANCE.value,
                priority=float(best.score),
                confidence=1.0,
                created_at=now,
                expires_at=now + FOCUS_TTL,
            )
        conf = max(0.0, min(1.0, 1.0 - 0.5 * best.uncertainty_gain))
        return FocusState(
            target_id=best.entity_id,
            focus_type=best.focus_type,
            priority=float(best.score),
            confidence=conf,
            created_at=now,
            expires_at=now + FOCUS_TTL,
        )


def focus_from_world(world: WorldState, log: ExperienceLog | None = None, now: float = 0.0) -> FocusState:
    return FocusEngine().select(world, log, now)
