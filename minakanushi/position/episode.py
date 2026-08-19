"""Episode-position encoder."""

from __future__ import annotations

from torch import Tensor, nn

from minakanushi.position.temporal import FourierScalarEncoder


class EpisodeEncoder(nn.Module):
    def __init__(self, dim: int, num_frequencies: int) -> None:
        super().__init__()
        self.encoder = FourierScalarEncoder(dim, num_frequencies, scale=1.0)

    def forward(self, episode_position: Tensor) -> Tensor:
        return self.encoder(episode_position)
