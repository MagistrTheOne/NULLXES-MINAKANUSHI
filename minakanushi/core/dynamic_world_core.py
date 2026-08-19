"""DynamicWorldCore — primary learned substrate.

S_{t+1} = DWC(S_t, O_{t+1}, M_t, P_t)

One external observation may execute multiple internal cognition cycles.
Kinematic readouts are first-class because the native domain is physical
state, not next-token prediction. Learned residual dynamics sit on top of
the kinematic prior.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from minakanushi.architecture.config import ArchitectureConfig
from minakanushi.architecture.mina_unit import MinaUnitBatch
from minakanushi.architecture.outputs import CoreOutput, PositionState
from minakanushi.core.cognitive_block import CognitiveBlock
from minakanushi.core.convergence import slot_delta
from minakanushi.core.recurrent_state import clone_world
from minakanushi.state.world import WorldState
from minakanushi.utils.tensors import assert_finite, assert_shape


class DynamicWorldCore(nn.Module):
    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.latent_dim
        self.blocks = nn.ModuleList([CognitiveBlock(config) for _ in range(config.core_depth)])
        self.obs_fuse = nn.Linear(dim * 2, dim)
        self.vel_head = nn.Linear(dim, 2)
        self.xy_residual = nn.Linear(dim, 2)
        self.memory_write = nn.Linear(dim, config.memory_dim)
        self.seed_head = nn.Linear(dim, dim)

    def forward(
        self,
        world_state: WorldState,
        observation_state: Tensor,
        memory_state: Tensor,
        position_state: PositionState,
        units: MinaUnitBatch,
        cognition_budget: int | None = None,
    ) -> CoreOutput:
        """
        observation_state: [B, N_obs, D] semantic embeddings
        memory_state:      [B, N_world, D] retrieved memory aligned to slots
        position_state.embedding: [B, N_obs, D]
        """
        budget = cognition_budget if cognition_budget is not None else self.config.cognition.budget
        if budget < 1:
            raise ValueError("cognition_budget must be >= 1")
        fused_obs = self.obs_fuse(torch.cat([observation_state, position_state.embedding], dim=-1))
        state = clone_world(world_state)
        if memory_state is not None:
            assert_shape("memory_state", memory_state, tuple(state.latent_state.shape))
            state.latent_state = state.latent_state + 0.1 * memory_state
        cycles = 0
        last_delta = torch.ones(state.latent_state.shape[0], device=state.latent_state.device)
        for _ in range(budget):
            previous = state.latent_state
            updated = previous
            for block in self.blocks:
                updated = block(updated, fused_obs, state.occupied, units.mask)
            last_delta = slot_delta(previous, updated, state.occupied)
            state.latent_state = updated
            cycles += 1
            if bool((last_delta < self.config.cognition.convergence_threshold).all()):
                break
        dt = self.config.dt
        vel = self.vel_head(state.latent_state)
        correction = self.xy_residual(state.latent_state)
        observed = (state.age_unobserved == 0) & state.occupied
        # Observed slots keep evidence position plus a learned noise correction.
        # Unobserved slots roll kinematics. Predictions never write back through FutureEngine.
        xy_now = torch.where(
            observed.unsqueeze(-1),
            state.entity_xy + correction,
            state.entity_xy + state.entity_vel * dt + correction,
        )
        inferred_vel = torch.where(observed.unsqueeze(-1), vel, state.entity_vel)
        state.entity_vel = torch.where(state.occupied.unsqueeze(-1), inferred_vel, torch.zeros_like(inferred_vel))
        state.entity_xy = torch.where(state.occupied.unsqueeze(-1), xy_now, torch.zeros_like(xy_now))
        assert_finite("world.latent_state", state.latent_state)
        writes = self.memory_write(state.latent_state)
        seed = self.seed_head(state.latent_state.mean(dim=1))
        return CoreOutput(
            world_state=state,
            memory_write_candidates=writes,
            uncertainty_state=state.uncertainty,
            prediction_seed=seed,
            convergence_score=last_delta,
            cognition_cycles=cycles,
        )
