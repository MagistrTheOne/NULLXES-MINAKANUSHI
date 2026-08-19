"""Vector observation encoder → MinaUnit semantic embeddings."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from minakanushi.architecture.config import ArchitectureConfig


class VectorEncoder(nn.Module):
    """Input features: [kind, x, y, vx, vy, confidence, noise] → [D]."""

    FEATURE_DIM = 7

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, config.latent_dim),
            nn.SiLU(),
            nn.Linear(config.latent_dim, config.latent_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[-1] != self.FEATURE_DIM:
            raise ValueError(f"vector features last dim must be {self.FEATURE_DIM}, got {tuple(features.shape)}")
        return self.net(features)
