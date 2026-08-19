"""PerceptionBridge — heterogeneous observations to MinaUnits.

The World Core never depends on camera resolution or sensor vendor. Milestone 1
encodes vector and telemetry observations. Vision and text are omitted until
implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from minakanushi.architecture.config import ArchitectureConfig
from minakanushi.architecture.mina_unit import KIND_IDS, MinaUnit, SOURCE_TYPES
from minakanushi.perception.telemetry import TelemetryEncoder
from minakanushi.perception.vector import VectorEncoder


@dataclass
class Observation:
    timestamp: float
    agent_xy: tuple[float, float]
    agent_vel: tuple[float, float]
    heading: float
    health: float
    battery: float
    visible: tuple[dict[str, Any], ...] = ()
    occluded_ids: tuple[int, ...] = ()
    noise_std: float = 0.0
    arrival_time: float | None = None
    source_rate_telemetry: float = 20.0
    source_rate_vector: float = 10.0
    metadata: dict[str, Any] = field(default_factory=dict)


class PerceptionBridge(nn.Module):
    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        self.vector = VectorEncoder(config)
        self.telemetry = TelemetryEncoder(config)

    def encode(self, observation: Observation, *, device: torch.device, dtype: torch.dtype) -> list[MinaUnit]:
        units: list[MinaUnit] = []
        tel = torch.tensor(
            [
                observation.agent_xy[0],
                observation.agent_xy[1],
                observation.agent_vel[0],
                observation.agent_vel[1],
                observation.heading,
                observation.health,
                observation.battery,
            ],
            device=device,
            dtype=dtype,
        )
        arrival = observation.arrival_time if observation.arrival_time is not None else observation.timestamp
        units.append(
            MinaUnit(
                source_type="telemetry",
                source_id=1,
                timestamp=float(observation.metadata.get("event_time", observation.timestamp)),
                sequence_index=0,
                spatial_frame="arena",
                spatial_position=(observation.agent_xy[0], observation.agent_xy[1], 0.0),
                spatial_valid=True,
                semantic_embedding=self.telemetry(tel.unsqueeze(0)).squeeze(0),
                confidence=0.98,
                uncertainty=0.02,
                persistence=1.0,
                entity_reference=1,
                relation_reference=0,
                kind="agent",
                arrival_time=arrival,
                source_rate=observation.source_rate_telemetry,
            )
        )
        for seq, item in enumerate(observation.visible, start=1):
            kind = str(item.get("kind", "unknown"))
            conf = float(item.get("confidence", 1.0))
            feat = torch.tensor(
                [
                    float(KIND_IDS.get(kind, 0)),
                    float(item["xy"][0]),
                    float(item["xy"][1]),
                    float(item.get("vel", (0.0, 0.0))[0]),
                    float(item.get("vel", (0.0, 0.0))[1]),
                    conf,
                    float(observation.noise_std),
                ],
                device=device,
                dtype=dtype,
            )
            event_t = float(item.get("event_time", observation.timestamp))
            arr_t = float(item.get("arrival_time", arrival))
            rate = float(item.get("source_rate", observation.source_rate_vector))
            units.append(
                MinaUnit(
                    source_type="vector",
                    source_id=2,
                    timestamp=event_t,
                    sequence_index=seq,
                    spatial_frame="arena",
                    spatial_position=(float(item["xy"][0]), float(item["xy"][1]), 0.0),
                    spatial_valid=True,
                    semantic_embedding=self.vector(feat.unsqueeze(0)).squeeze(0),
                    confidence=conf,
                    uncertainty=max(0.0, 1.0 - conf),
                    persistence=0.8,
                    entity_reference=int(item["id"]),
                    relation_reference=0,
                    kind=kind,
                    arrival_time=arr_t,
                    source_rate=rate,
                    metadata={"occluded": "false"},
                )
            )
        return units

    def encode_feature_batch(self, features: Tensor, source: str) -> Tensor:
        if source == "vector":
            return self.vector(features)
        if source == "telemetry":
            return self.telemetry(features)
        raise ValueError(f"unsupported perception source '{source}'")
