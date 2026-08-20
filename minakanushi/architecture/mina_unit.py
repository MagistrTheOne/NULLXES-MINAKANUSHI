"""MinaUnit — native information carrier.

A MinaUnit is a temporally and optionally spatially grounded piece of
information. Text tokens are not privileged over physical observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from minakanushi.utils.tensors import assert_finite, assert_shape


SOURCE_TYPES: dict[str, int] = {
    "vector": 0,
    "telemetry": 1,
    "image": 2,
    "structured_event": 3,
    "text": 4,
    "system_state": 5,
    "memory": 6,
    "prediction": 7,
}

KIND_IDS: dict[str, int] = {
    "unknown": 0,
    "agent": 1,
    "mover": 2,
    "obstacle": 3,
    "target": 4,
    "zone": 5,
    "event": 6,
}


@dataclass
class MinaUnit:
    source_type: str
    source_id: int
    timestamp: float
    sequence_index: int
    spatial_frame: str
    spatial_position: tuple[float, float, float]
    spatial_valid: bool
    semantic_embedding: Tensor
    confidence: float
    uncertainty: float
    persistence: float
    entity_reference: int
    relation_reference: int
    causal_parent_ids: tuple[int, ...] = ()
    kind: str = "unknown"
    arrival_time: float | None = None
    source_rate: float = 10.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # timestamp is event_time. arrival_time is processor arrival.
        # Equal values means zero sensor delay, not "time is one scalar".
        if self.arrival_time is None:
            self.arrival_time = self.timestamp

    def validate(self) -> None:
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(f"unknown source_type '{self.source_type}'")
        if self.kind not in KIND_IDS:
            raise ValueError(f"unknown kind '{self.kind}'")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence out of range: {self.confidence}")
        if self.uncertainty < 0.0:
            raise ValueError(f"uncertainty must be >= 0, got {self.uncertainty}")
        assert_shape("MinaUnit.semantic_embedding", self.semantic_embedding, (self.semantic_embedding.numel(),))
        assert_finite("MinaUnit.semantic_embedding", self.semantic_embedding)


@dataclass
class MinaUnitBatch:
    """Batched MinaUnits for learned modules.

    semantic_embedding: [B, N, D]  float — unit semantics before NPF fusion
    timestamp:          [B, N]     float — event_time, seconds (when the event occurred)
    arrival_time:       [B, N]     float — processor arrival time; may differ from event_time
    source_rate:        [B, N]     float — originating stream sampling rate (Hz)
    sequence_index:     [B, N]     long  — order inside one observation stream
    spatial_position:   [B, N, 3]  float — coordinates in declared frame
    spatial_valid:      [B, N]     bool  — False if the unit has no spatial meaning
    episode_position:   [B, N]     float — position inside the current episode
    memory_age:         [B, N]     float — now - event_time, seconds
    source_id:          [B, N]     long  — originating stream id
    source_type:        [B, N]     long  — SOURCE_TYPES
    confidence:         [B, N]     float
    uncertainty:        [B, N]     float
    entity_id:          [B, N]     long
    kind:               [B, N]     long
    mask:               [B, N]     bool  — True = valid unit
    velocity:           [B, N, 2]  float — reported planar velocity if the source has it
    """

    semantic_embedding: Tensor
    timestamp: Tensor
    arrival_time: Tensor
    source_rate: Tensor
    sequence_index: Tensor
    spatial_position: Tensor
    spatial_valid: Tensor
    episode_position: Tensor
    memory_age: Tensor
    source_id: Tensor
    source_type: Tensor
    confidence: Tensor
    uncertainty: Tensor
    entity_id: Tensor
    kind: Tensor
    mask: Tensor
    velocity: Tensor

    def validate(self) -> None:
        batch, count, dim = self.semantic_embedding.shape
        assert_shape("semantic_embedding", self.semantic_embedding, (batch, count, dim))
        assert_shape("timestamp", self.timestamp, (batch, count))
        assert_shape("arrival_time", self.arrival_time, (batch, count))
        assert_shape("source_rate", self.source_rate, (batch, count))
        assert_shape("sequence_index", self.sequence_index, (batch, count))
        assert_shape("spatial_position", self.spatial_position, (batch, count, 3))
        assert_shape("spatial_valid", self.spatial_valid, (batch, count))
        assert_shape("episode_position", self.episode_position, (batch, count))
        assert_shape("memory_age", self.memory_age, (batch, count))
        assert_shape("source_id", self.source_id, (batch, count))
        assert_shape("kind", self.kind, (batch, count))
        assert_shape("mask", self.mask, (batch, count))
        assert_shape("velocity", self.velocity, (batch, count, 2))
        assert_finite("semantic_embedding", self.semantic_embedding)
        assert_finite("timestamp", self.timestamp)


def empty_batch(
    batch_size: int,
    max_units: int,
    latent_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> MinaUnitBatch:
    z = torch.zeros(batch_size, max_units, latent_dim, device=device, dtype=dtype)
    n = torch.zeros(batch_size, max_units, device=device, dtype=dtype)
    l = torch.zeros(batch_size, max_units, device=device, dtype=torch.long)
    b = torch.zeros(batch_size, max_units, device=device, dtype=torch.bool)
    return MinaUnitBatch(
        semantic_embedding=z,
        timestamp=n.clone(),
        arrival_time=n.clone(),
        source_rate=torch.ones(batch_size, max_units, device=device, dtype=dtype),
        sequence_index=l.clone(),
        spatial_position=torch.zeros(batch_size, max_units, 3, device=device, dtype=dtype),
        spatial_valid=b.clone(),
        episode_position=n.clone(),
        memory_age=n.clone(),
        source_id=l.clone(),
        source_type=l.clone(),
        confidence=n.clone(),
        uncertainty=n.clone(),
        entity_id=l.clone(),
        kind=l.clone(),
        mask=b.clone(),
        velocity=torch.zeros(batch_size, max_units, 2, device=device, dtype=dtype),
    )


def pack_units(
    units: list[MinaUnit],
    *,
    batch_index: int,
    max_units: int,
    latent_dim: int,
    episode_position: float,
    now: float,
    device: torch.device,
    dtype: torch.dtype,
) -> MinaUnitBatch:
    packed = empty_batch(1, max_units, latent_dim, device=device, dtype=dtype)
    if batch_index != 0:
        raise ValueError("pack_units currently emits B=1; use cat_batches for training")
    count = min(len(units), max_units)
    for i in range(count):
        unit = units[i]
        unit.validate()
        if unit.semantic_embedding.numel() != latent_dim:
            raise ValueError(
                f"MinaUnit.semantic_embedding dim {unit.semantic_embedding.numel()} != latent_dim {latent_dim}"
            )
        packed.semantic_embedding[0, i] = unit.semantic_embedding.to(device=device, dtype=dtype)
        packed.timestamp[0, i] = unit.timestamp
        packed.arrival_time[0, i] = float(unit.arrival_time)
        packed.source_rate[0, i] = float(unit.source_rate)
        packed.sequence_index[0, i] = unit.sequence_index
        packed.spatial_position[0, i] = torch.tensor(unit.spatial_position, device=device, dtype=dtype)
        packed.spatial_valid[0, i] = unit.spatial_valid
        packed.episode_position[0, i] = episode_position
        packed.memory_age[0, i] = max(0.0, now - unit.timestamp)
        packed.source_id[0, i] = unit.source_id
        packed.source_type[0, i] = SOURCE_TYPES[unit.source_type]
        packed.confidence[0, i] = unit.confidence
        packed.uncertainty[0, i] = unit.uncertainty
        packed.entity_id[0, i] = unit.entity_reference
        packed.kind[0, i] = KIND_IDS[unit.kind]
        packed.mask[0, i] = True
        vel = unit.metadata.get("vel", (0.0, 0.0))
        packed.velocity[0, i, 0] = float(vel[0])
        packed.velocity[0, i, 1] = float(vel[1])
    packed.validate()
    return packed
