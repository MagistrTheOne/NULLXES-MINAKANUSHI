"""ExperienceRecord — structured operational history. Not chat, not RAG.

Experience is:
  situation, prediction, reality, error, correction, lesson

It is not store-tensor / retrieve-tensor.
"""

from __future__ import annotations

from dataclasses import dataclass, field


LESSON_CONSISTENT = "consistent"
LESSON_VELOCITY = "velocity_discontinuity"
LESSON_POSITION = "position_surprise"
LESSON_REVISION = "hypothesis_revision"


def _pair(raw: object, default: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return (float(raw[0]), float(raw[1]))
    return default


@dataclass(frozen=True)
class ExperienceRecord:
    event_time: float
    situation: str
    action: str
    entity_id: int = 0
    predicted_xy: tuple[float, float] = (0.0, 0.0)
    predicted_vel: tuple[float, float] = (0.0, 0.0)
    observed_xy: tuple[float, float] = (0.0, 0.0)
    observed_vel: tuple[float, float] = (0.0, 0.0)
    error_xy: float = 0.0
    error_vel: float = 0.0
    correction_required: bool = False
    correction_reason: str = ""
    lesson: str = LESSON_CONSISTENT
    # Gate 04 aliases kept for checkpoint roundtrip of old logs.
    belief_before: str = ""
    belief_after: str = ""
    result: str = ""

    def to_dict(self) -> dict:
        return {
            "event_time": self.event_time,
            "situation": self.situation,
            "action": self.action,
            "entity_id": self.entity_id,
            "predicted_xy": list(self.predicted_xy),
            "predicted_vel": list(self.predicted_vel),
            "observed_xy": list(self.observed_xy),
            "observed_vel": list(self.observed_vel),
            "error_xy": self.error_xy,
            "error_vel": self.error_vel,
            "correction_required": self.correction_required,
            "correction_reason": self.correction_reason,
            "lesson": self.lesson,
            "belief_before": self.belief_before or str(self.predicted_xy),
            "belief_after": self.belief_after or str(self.observed_xy),
            "result": self.result or self.lesson,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> ExperienceRecord:
        return cls(
            event_time=float(raw["event_time"]),
            situation=str(raw.get("situation", "")),
            action=str(raw.get("action", "")),
            entity_id=int(raw.get("entity_id", 0)),
            predicted_xy=_pair(raw.get("predicted_xy")),
            predicted_vel=_pair(raw.get("predicted_vel")),
            observed_xy=_pair(raw.get("observed_xy")),
            observed_vel=_pair(raw.get("observed_vel")),
            error_xy=float(raw.get("error_xy", 0.0)),
            error_vel=float(raw.get("error_vel", 0.0)),
            correction_required=bool(raw.get("correction_required", False)),
            correction_reason=str(raw.get("correction_reason", "")),
            lesson=str(raw.get("lesson", LESSON_CONSISTENT)),
            belief_before=str(raw.get("belief_before", "")),
            belief_after=str(raw.get("belief_after", "")),
            result=str(raw.get("result", "")),
        )


@dataclass
class ExperienceLog:
    records: list[ExperienceRecord] = field(default_factory=list)
    cap: int = 64

    def append(self, record: ExperienceRecord) -> None:
        self.records.append(record)
        if len(self.records) > self.cap:
            self.records = self.records[-self.cap :]

    def latest_for(self, entity_id: int) -> ExperienceRecord | None:
        for rec in reversed(self.records):
            if rec.entity_id == entity_id:
                return rec
        return None

    def to_dict(self) -> dict:
        return {"cap": self.cap, "records": [r.to_dict() for r in self.records]}

    @classmethod
    def from_dict(cls, raw: dict) -> ExperienceLog:
        log = cls(cap=int(raw.get("cap", 64)))
        log.records = [ExperienceRecord.from_dict(item) for item in raw.get("records", [])]
        return log
