"""Memory engine: working + episodic retrieval for the cognitive loop.

Runtime store detaches (bounded buffer). Training must pass live write
candidates from a previous step as `live_writes` so L_memory can reach
memory_write and the DWC update that consumes retrieval.
"""

from __future__ import annotations

from torch import Tensor, nn

from minakanushi.architecture.config import ArchitectureConfig
from minakanushi.memory.episodic import EpisodicMemory
from minakanushi.memory.working import WorkingMemory
from minakanushi.state.world import WorldState
from minakanushi.utils.tensors import assert_shape


class MemoryEngine(nn.Module):
    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        self.working = WorkingMemory(config)
        self.episodic = EpisodicMemory(config)
        self.read_proj = nn.Linear(config.memory_dim, config.latent_dim)

    def hints(self, world: WorldState, live_writes: Tensor | None = None) -> Tensor:
        """Return [B, N_world, D] that participates in StateConstructor and DWC."""
        if live_writes is not None:
            assert_shape("live_writes", live_writes, tuple(world.latent_state.shape))
            retrieved = self.read_proj(live_writes)
        else:
            retrieved = self.read_proj(self.episodic.retrieve_for_slots(world))
        working = self.working.readout(world.latent_state.shape[1], world.latent_state.shape[2])
        return retrieved + 0.2 * working

    def write(self, world: WorldState, candidates: Tensor) -> None:
        occupied = world.occupied[0]
        if bool(occupied.any()):
            pooled = world.latent_state[0, occupied].mean(dim=0, keepdim=True)
        else:
            pooled = world.latent_state.mean(dim=1)
        self.working.write(pooled)
        self.episodic.write_world(world, candidates)
