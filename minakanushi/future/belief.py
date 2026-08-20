"""Action-conditioned belief transition.

Contract: Belief(t) + Action(t) → Belief(t+1)

This is not state→future_xy and not observation→next observation.
Possibility must not write back into current WorldState.
"""

from __future__ import annotations

import torch

from minakanushi.core.recurrent_state import clone_world
from minakanushi.state.entity import AGENT_SLOT
from minakanushi.state.world import (
    BELIEF_EXISTENCE_FLOOR,
    BELIEF_STD_MIN,
    COAST_STD_GAIN,
    EXISTENCE_DECAY,
    WorldState,
)
from minakanushi.strategy.candidate import StrategyCandidate
from minakanushi.strategy.hold import is_hold


def action_plant_velocity(strategy: StrategyCandidate, agent_xy: torch.Tensor, speed: float = 1.0) -> torch.Tensor:
    """Directed plant velocity for the agent slot. Holds are zero."""
    zeros = torch.zeros_like(agent_xy)
    if is_hold(strategy.objective):
        return zeros
    target = torch.tensor(strategy.target_xy, device=agent_xy.device, dtype=agent_xy.dtype).unsqueeze(0)
    delta = target - agent_xy
    norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1e-6)
    return (delta / norm) * speed


def roll_belief(
    world: WorldState,
    strategy: StrategyCandidate,
    *,
    steps: int,
    dt: float,
) -> WorldState:
    """Kinematic future belief. Non-agent slots coast; only the agent takes the action."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    state = clone_world(world)
    agent_xy = state.entity_xy[:, AGENT_SLOT]
    plant = action_plant_velocity(strategy, agent_xy)
    occ = state.occupied.unsqueeze(-1).to(state.entity_xy.dtype)
    for _ in range(steps):
        vel = state.entity_vel.clone()
        vel[:, AGENT_SLOT] = plant
        state.entity_vel = torch.where(state.occupied.unsqueeze(-1), vel, torch.zeros_like(vel))
        state.entity_xy = state.entity_xy + state.entity_vel * dt * occ
        state.xy_std = (state.xy_std + COAST_STD_GAIN * 0.15).clamp_min(BELIEF_STD_MIN)
        state.vel_std = (state.vel_std + COAST_STD_GAIN * 0.1).clamp_min(BELIEF_STD_MIN)
        state.existence = (state.existence * EXISTENCE_DECAY).clamp(min=BELIEF_EXISTENCE_FLOOR)
        state.existence = torch.where(state.occupied, state.existence, torch.zeros_like(state.existence))
        state.pred_confidence = (state.pred_confidence * 0.95).clamp(min=1e-4)
        state.age_unobserved = state.age_unobserved + state.occupied.to(state.age_unobserved.dtype)
        state.timestamp = state.timestamp + dt
    state.provenance = "future_belief"
    return state
