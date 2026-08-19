"""Spatial encoder. Missing coordinates must not be hallucinated."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SpatialEncoder(nn.Module):
    """shape in: position [B, N, 3], valid [B, N]
    shape out: [B, N, D]
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, spatial_position: Tensor, spatial_valid: Tensor) -> Tensor:
        if spatial_position.shape[-1] != 3:
            raise ValueError(f"spatial_position last dim must be 3, got {tuple(spatial_position.shape)}")
        encoded = self.mlp(spatial_position)
        gate = spatial_valid.to(encoded.dtype).unsqueeze(-1)
        return encoded * gate
