"""Source-stream encoder. Source identity is not semantics."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SourceEncoder(nn.Module):
    def __init__(self, dim: int, max_sources: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(max_sources, dim)

    def forward(self, source_id: Tensor) -> Tensor:
        if source_id.dtype not in (torch.int32, torch.int64, torch.long):
            raise ValueError(f"source_id must be integer ids, got {source_id.dtype}")
        max_id = int(self.embed.num_embeddings) - 1
        if bool((source_id < 0).any() or (source_id > max_id).any()):
            raise ValueError(f"source_id out of range [0, {max_id}]")
        return self.embed(source_id)
