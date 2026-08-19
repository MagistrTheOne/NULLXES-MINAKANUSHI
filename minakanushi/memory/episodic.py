"""Episodic memory of (S, A, S') transitions with entity links."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from minakanushi.architecture.config import ArchitectureConfig
from minakanushi.state.world import WorldState


@dataclass
class MemoryEntry:
    created_at: float
    state_embedding: Tensor
    importance: float
    confidence: float
    entity_id: int
    xy: tuple[float, float]


class EpisodicMemory(nn.Module):
    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        self.slots = config.memory_slots
        self.register_buffer("embeddings", torch.zeros(self.slots, config.memory_dim), persistent=True)
        self.register_buffer("entity_ids", torch.zeros(self.slots, dtype=torch.long), persistent=True)
        self.register_buffer("xy", torch.zeros(self.slots, 2), persistent=True)
        self.register_buffer("importance", torch.zeros(self.slots), persistent=True)
        self.register_buffer("valid", torch.zeros(self.slots, dtype=torch.bool), persistent=True)
        self.write_index = 0
        self.reads = 0
        self.writes = 0

    def write_world(self, world: WorldState, candidates: Tensor) -> int:
        """Store occupied slots. candidates: [B, N_world, memory_dim]."""
        written = 0
        occupied = world.occupied[0]
        for slot in occupied.nonzero(as_tuple=False).flatten().tolist():
            idx = self.write_index % self.slots
            self.embeddings[idx] = candidates[0, slot].detach()
            self.entity_ids[idx] = world.entity_id[0, slot]
            self.xy[idx] = world.entity_xy[0, slot].detach()
            self.importance[idx] = world.confidence[0, slot].detach()
            self.valid[idx] = True
            self.write_index += 1
            written += 1
        self.writes += written
        return written

    def retrieve_for_slots(self, world: WorldState) -> Tensor:
        """Return [B, N_world, D] hints keyed by entity_id."""
        hints = torch.zeros_like(world.latent_state)
        if not bool(self.valid.any()):
            return hints
        for b in range(world.entity_id.shape[0]):
            for s in range(world.entity_id.shape[1]):
                if not bool(world.occupied[b, s]):
                    continue
                eid = world.entity_id[b, s]
                matches = self.valid & (self.entity_ids == eid)
                if not bool(matches.any()):
                    continue
                idxs = matches.nonzero(as_tuple=False).flatten()
                hints[b, s] = self.embeddings[idxs].mean(dim=0)[: world.latent_state.shape[-1]]
                self.reads += 1
        return hints
