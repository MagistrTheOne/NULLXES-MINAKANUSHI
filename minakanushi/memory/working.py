"""Working memory — high-resolution current context."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from minakanushi.architecture.config import ArchitectureConfig


class WorkingMemory(nn.Module):
    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.capacity = min(8, config.memory_slots)
        self.register_buffer("buffer", torch.zeros(1, self.capacity, config.memory_dim), persistent=True)
        self.register_buffer("filled", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("cursor", torch.zeros((), dtype=torch.long), persistent=True)

    def write(self, state_embedding: Tensor) -> None:
        """state_embedding: [B, D] pooled world embedding. Stored without graph."""
        idx = int(self.cursor.item())
        self.buffer[0, idx] = state_embedding[0].detach()
        self.cursor.fill_((idx + 1) % self.capacity)
        self.filled.fill_(min(int(self.filled.item()) + 1, self.capacity))

    def readout(self, slots: int, dim: int) -> Tensor:
        filled = int(self.filled.item())
        if filled == 0:
            return torch.zeros(1, slots, dim, device=self.buffer.device, dtype=self.buffer.dtype)
        pooled = self.buffer[0, :filled].mean(dim=0)
        return pooled.view(1, 1, -1).expand(1, slots, dim)
