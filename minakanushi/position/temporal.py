"""Fourier features over a scalar physical coordinate."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class FourierScalarEncoder(nn.Module):
    """Maps a scalar to D dimensions with log-spaced sinusoids plus a linear skip.

    shape in:  [B, N]
    shape out: [B, N, D]
    """

    def __init__(self, dim: int, num_frequencies: int, scale: float = 1.0) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("encoder dim must be >= 2")
        self.dim = dim
        self.num_frequencies = num_frequencies
        self.scale = scale
        frequencies = scale * (2.0 ** torch.linspace(0.0, num_frequencies - 1, num_frequencies))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.proj = nn.Linear(2 * num_frequencies + 1, dim)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 2:
            raise ValueError(f"expected [B, N], got {tuple(values.shape)}")
        scaled = values.unsqueeze(-1) * self.frequencies * (2.0 * math.pi)
        features = torch.cat([values.unsqueeze(-1), torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return self.proj(features)


class PhysicalTimeEncoder(nn.Module):
    """Encodes event_time, arrival_time, delay, and source_rate into one D-vector.

    event_time != arrival_time is a first-class distinction.
    shape in:  each [B, N]
    shape out: [B, N, D]
    """

    def __init__(self, dim: int, num_frequencies: int, scale: float = 1.0) -> None:
        super().__init__()
        self.event = FourierScalarEncoder(dim, num_frequencies, scale=scale)
        self.arrival = FourierScalarEncoder(dim, num_frequencies, scale=scale)
        self.delay = FourierScalarEncoder(dim, num_frequencies, scale=scale)
        self.rate = FourierScalarEncoder(dim, num_frequencies, scale=0.1)
        self.fuse = nn.Linear(dim * 4, dim)

    def forward(self, event_time: Tensor, arrival_time: Tensor, source_rate: Tensor) -> Tensor:
        delay = (arrival_time - event_time).clamp_min(0.0)
        fused = torch.cat(
            [
                self.event(event_time),
                self.arrival(arrival_time),
                self.delay(delay),
                self.rate(source_rate.clamp_min(0.0)),
            ],
            dim=-1,
        )
        return self.fuse(fused)
