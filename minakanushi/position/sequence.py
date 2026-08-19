"""Sequence-order encoder inside one observation stream."""

from __future__ import annotations

from torch import Tensor, nn

from minakanushi.position.temporal import FourierScalarEncoder


class SequenceEncoder(nn.Module):
    def __init__(self, dim: int, num_frequencies: int) -> None:
        super().__init__()
        self.encoder = FourierScalarEncoder(dim, num_frequencies, scale=1.0)

    def forward(self, sequence_index: Tensor) -> Tensor:
        return self.encoder(sequence_index.to(dtype=self.encoder.proj.weight.dtype))
