"""Recurrent world-state container helpers."""

from __future__ import annotations

from minakanushi.state.world import WorldState


def clone_world(state: WorldState) -> WorldState:
    return WorldState(
        timestamp=state.timestamp.clone(),
        latent_state=state.latent_state.clone(),
        entity_xy=state.entity_xy.clone(),
        entity_vel=state.entity_vel.clone(),
        occupied=state.occupied.clone(),
        entity_id=state.entity_id.clone(),
        kind=state.kind.clone(),
        confidence=state.confidence.clone(),
        uncertainty=state.uncertainty.clone(),
        age_unobserved=state.age_unobserved.clone(),
        self_index=state.self_index,
        provenance=state.provenance,
    )
