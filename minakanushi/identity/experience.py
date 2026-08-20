"""ExperienceRecord — structured operational history. Not chat, not RAG."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class ExperienceRecord:
    event_time: float
    situation: str
    belief_before: str
    action: str
    result: str
    belief_after: str
    correction_required: bool

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> ExperienceRecord:
        return cls(
            event_time=float(raw["event_time"]),
            situation=str(raw["situation"]),
            belief_before=str(raw["belief_before"]),
            action=str(raw["action"]),
            result=str(raw["result"]),
            belief_after=str(raw["belief_after"]),
            correction_required=bool(raw["correction_required"]),
        )


@dataclass
class ExperienceLog:
    records: list[ExperienceRecord] = field(default_factory=list)
    cap: int = 64

    def append(self, record: ExperienceRecord) -> None:
        self.records.append(record)
        if len(self.records) > self.cap:
            self.records = self.records[-self.cap :]

    def to_dict(self) -> dict:
        return {"cap": self.cap, "records": [r.to_dict() for r in self.records]}

    @classmethod
    def from_dict(cls, raw: dict) -> ExperienceLog:
        log = cls(cap=int(raw.get("cap", 64)))
        log.records = [ExperienceRecord.from_dict(item) for item in raw.get("records", [])]
        return log
