"""Structured outputs crossing MINAKANUSHI subsystem boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor


@dataclass
class PositionState:
    """NPF output.

    embedding:           [B, N, D] fused positional state
    temporal_embedding:  [B, N, D] physical-time encoder
    spatial_embedding:   [B, N, D] spatial encoder (zero if invalid)
    episode_embedding:   [B, N, D]
    memory_embedding:    [B, N, D] memory-age encoder
    source_embedding:    [B, N, D]
    sequence_embedding:  [B, N, D]
    """

    embedding: Tensor
    temporal_embedding: Tensor
    spatial_embedding: Tensor
    episode_embedding: Tensor
    memory_embedding: Tensor
    source_embedding: Tensor
    sequence_embedding: Tensor


@dataclass
class CoreOutput:
    world_state: Any
    memory_write_candidates: Tensor
    uncertainty_state: Any
    prediction_seed: Tensor
    convergence_score: Tensor
    cognition_cycles: int


@dataclass
class CycleTelemetry:
    cycle_id: int
    physical_time: float
    observation_count: int
    entity_count: int
    event_count: int
    world_state_confidence: float
    uncertainty: float
    memory_reads: int
    memory_writes: int
    future_branches: int
    candidate_strategies: int
    rejected_strategies: int
    rejection_reasons: tuple[str, ...]
    selected_strategy: str
    cognition_cycles: int
    latency_ms: float
    extras: dict[str, Any] = field(default_factory=dict)
