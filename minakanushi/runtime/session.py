"""Persistent runtime session state."""

from __future__ import annotations

from dataclasses import dataclass, field

from minakanushi.causal.graph import CausalGraph
from minakanushi.identity.authority import AuthorityModel
from minakanushi.identity.focus import FocusState
from minakanushi.identity.persona import PersonaModel
from minakanushi.identity.self_model import SelfModel
from minakanushi.policy.intent import ActionIntent
from minakanushi.state.world import WorldState


@dataclass
class SessionState:
    cycle_id: int
    episode_position: float
    world: WorldState
    last_intent: ActionIntent | None = None
    causal: CausalGraph = field(default_factory=CausalGraph)
    self_model: SelfModel | None = None
    authority: AuthorityModel | None = None
    persona: PersonaModel | None = None
    focus: FocusState | None = None
