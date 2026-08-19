"""Causal edge — hypothesized transition, not mere correlation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CausalEdge:
    source_event: str
    target_event: str
    relation_type: str
    confidence: float
    temporal_delay: float
    evidence_count: int
    contradictory_evidence: int
