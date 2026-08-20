"""Structured multimodal observation. Not RGB. Not tokens."""

from __future__ import annotations

from dataclasses import dataclass, field

from minakanushi.perception.bridge import Observation

AUDIO_KINDS = ("fall", "shout", "impact", "alarm", "voice_command")

VISION_KIND_MAP = {
    "human": "mover",
    "vehicle": "mover",
    "obstacle": "obstacle",
    "target": "target",
    "mover": "mover",
    "zone": "zone",
    "event": "event",
    "agent": "agent",
}


@dataclass(frozen=True)
class VisionDetection:
    id: int
    type: str
    xy: tuple[float, float]
    vel: tuple[float, float] = (0.0, 0.0)
    relation: str = ""
    event: str = ""
    confidence: float = 1.0


@dataclass(frozen=True)
class AudioEvent:
    kind: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in AUDIO_KINDS:
            raise ValueError(f"unknown audio kind {self.kind!r}")


@dataclass(frozen=True)
class OperatorIntent:
    target: str
    constraint: str
    priority: str
    utterance: str = ""


@dataclass(frozen=True)
class EmbodimentState:
    battery: float = 1.0
    joints: tuple[float, ...] = ()
    force: float = 0.0
    balance: float = 1.0


@dataclass
class MultimodalObservation:
    base: Observation
    detections: tuple[VisionDetection, ...] = ()
    audio: tuple[AudioEvent, ...] = ()
    operator: OperatorIntent | str | None = None
    embodiment: EmbodimentState | None = field(default=None)
