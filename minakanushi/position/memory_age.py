"""Memory-age encoder.

age_i = t_current - t_i  (seconds)
Relevance decays with age; the encoder must see log-time, not token index.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from minakanushi.position.temporal import FourierScalarEncoder


class MemoryAgeEncoder(nn.Module):
    def __init__(self, dim: int, num_frequencies: int) -> None:
        super().__init__()
        self.encoder = FourierScalarEncoder(dim, num_frequencies, scale=0.5)

    def forward(self, memory_age: Tensor) -> Tensor:
        return self.encoder(torch.log1p(memory_age.clamp_min(0.0)))
