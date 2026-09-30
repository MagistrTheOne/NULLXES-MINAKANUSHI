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
        hints, _ = self.hints_with_stats(world, live_writes=live_writes)
        return hints

    def hints_with_stats(
        self, world: WorldState, live_writes: Tensor | None = None
    ) -> tuple[Tensor, dict]:
        """Hints plus Gate-E instrumentation. No retrieval semantics change.

        Stats (diagnostic only, does not gate fusion):
          hit_count / miss_count / hit_ids / mean_age_hit /
          hint_norm / world_norm / live_path(bool)
        Use to prove the source of negative memΔ before any redesign.
        """
        if live_writes is not None:
            assert_shape("live_writes", live_writes, tuple(world.latent_state.shape))
            retrieved = self.read_proj(live_writes)
            live_path = True
        else:
            retrieved = self.read_proj(self.episodic.retrieve_for_slots(world))
            live_path = False
        working = self.working.readout(world.latent_state.shape[1], world.latent_state.shape[2])
        hints = retrieved + 0.2 * working
        stats: dict = {"live_path": live_path}
        try:
            occ = world.occupied[0]
            slots = occ.nonzero(as_tuple=False).flatten().tolist()
            valid = self.episodic.valid
            eids = self.episodic.entity_ids
            hit_ids: list[int] = []
            ages: list[float] = []
            miss = 0
            for s in slots:
                eid = int(world.entity_id[0, s].item())
                matches = valid & (eids == eid)
                if bool(matches.any()):
                    hit_ids.append(eid)
                    ages.append(float(world.age_unobserved[0, s].item()))
                else:
                    miss += 1
            stats.update(
                {
                    "hit_count": len(hit_ids),
                    "miss_count": miss,
                    "hit_ids": hit_ids,
                    "mean_age_hit": (sum(ages) / len(ages)) if ages else 0.0,
                    "hint_norm": float(hints.detach().float().norm(2).item()),
                    "world_norm": float(world.latent_state.detach().float().norm(2).item()),
                    "retrieved_norm": float(retrieved.detach().float().norm(2).item()),
                }
            )
        except (RuntimeError, ValueError, AttributeError):
            stats.update({"hit_count": -1, "miss_count": -1})
        return hints, stats

    def write(self, world: WorldState, candidates: Tensor) -> None:
        occupied = world.occupied[0]
        if bool(occupied.any()):
            pooled = world.latent_state[0, occupied].mean(dim=0, keepdim=True)
        else:
            pooled = world.latent_state.mean(dim=1)
        self.working.write(pooled)
        self.episodic.write_world(world, candidates)
