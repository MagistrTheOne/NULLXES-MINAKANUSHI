"""Persistent runtime session state."""

from __future__ import annotations

from dataclasses import dataclass, field

from minakanushi.causal.graph import CausalGraph
from minakanushi.policy.intent import ActionIntent
from minakanushi.state.world import WorldState


@dataclass
class SessionState:
    cycle_id: int
    episode_position: float
    world: WorldState
    last_intent: ActionIntent | None = None
    causal: CausalGraph = field(default_factory=CausalGraph)
