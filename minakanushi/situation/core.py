"""Situation Core — what the current world configuration means for the system."""

from __future__ import annotations

from dataclasses import dataclass

from minakanushi.architecture.mina_unit import KIND_IDS
from minakanushi.state.entity import AGENT_SLOT
from minakanushi.state.world import WorldState
from minakanushi.uncertainty.engine import UncertaintyState


@dataclass
class SituationState:
    world_state: WorldState
    relevant_entities: tuple[int, ...]
    active_events: tuple[str, ...]
    opportunities: tuple[str, ...]
    hazards: tuple[str, ...]
    goals: tuple[str, ...]
    constraints: tuple[str, ...]
    uncertainty: float
    causal_context: tuple[str, ...]
    temporal_context: str


class SituationCore:
    def build(self, world: WorldState, uncertainty: UncertaintyState, events: tuple[str, ...]) -> SituationState:
        agent_xy = world.entity_xy[0, AGENT_SLOT]
        movers = []
        targets = []
        hazards = []
        for slot in world.occupied[0].nonzero(as_tuple=False).flatten().tolist():
            eid = int(world.entity_id[0, slot].item())
            kind = int(world.kind[0, slot].item())
            if kind == KIND_IDS["mover"]:
                movers.append(eid)
            elif kind == KIND_IDS["target"]:
                targets.append(eid)
            elif kind == KIND_IDS["obstacle"]:
                hazards.append(f"obstacle:{eid}")
            dist = float((world.entity_xy[0, slot] - agent_xy).pow(2).sum().sqrt().item())
            if dist < 0.8 and slot != AGENT_SLOT:
                hazards.append(f"proximity:{eid}")
        mean_u = float(uncertainty.state_uncertainty[0, world.occupied[0]].mean().item()) if bool(world.occupied.any()) else 1.0
        goals = tuple(f"reach_target:{tid}" for tid in targets) or ("observe",)
        return SituationState(
            world_state=world,
            relevant_entities=tuple(movers + targets),
            active_events=events,
            opportunities=tuple(f"target:{tid}" for tid in targets),
            hazards=tuple(hazards),
            goals=goals,
            constraints=("hard_human_constraints",),
            uncertainty=mean_u,
            causal_context=events,
            temporal_context=f"t={float(world.timestamp[0].item()):.3f}",
        )
