"""Mutable causal graph over world events."""

from __future__ import annotations

from minakanushi.causal.edge import CausalEdge
from minakanushi.state.event import WorldEvent


class CausalGraph:
    def __init__(self) -> None:
        self.edges: list[CausalEdge] = []
        self.events: list[WorldEvent] = []

    def observe_transition(self, source: str, target: str, delay: float, consistent: bool) -> CausalEdge:
        for edge in self.edges:
            if edge.source_event == source and edge.target_event == target:
                evidence = edge.evidence_count + int(consistent)
                contra = edge.contradictory_evidence + int(not consistent)
                total = evidence + contra
                edge.confidence = evidence / max(total, 1)
                edge.evidence_count = evidence
                edge.contradictory_evidence = contra
                edge.temporal_delay = 0.8 * edge.temporal_delay + 0.2 * delay
                return edge
        edge = CausalEdge(source, target, "precedes", 1.0 if consistent else 0.0, delay, int(consistent), int(not consistent))
        self.edges.append(edge)
        return edge

    def record_event(self, event: WorldEvent) -> None:
        self.events.append(event)
