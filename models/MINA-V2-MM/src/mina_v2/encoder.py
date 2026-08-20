"""Small MLP: organ features → latent_dim. Lives in the V2 pack, not in frozen 1.0."""

from __future__ import annotations

from torch import Tensor, nn

from minakanushi.architecture.config import ArchitectureConfig


class OrganEncoder(nn.Module):
    def __init__(self, feature_dim: int, config: ArchitectureConfig) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, config.latent_dim),
            nn.SiLU(),
            nn.Linear(config.latent_dim, config.latent_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[-1] != self.feature_dim:
            raise ValueError(f"organ features last dim must be {self.feature_dim}, got {tuple(features.shape)}")
        return self.net(features)
