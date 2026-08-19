"""Relation tensor between world slots.

relation_logits: [B, N_world, N_world]
meaning: hypothesized interaction strength from i to j
"""

from __future__ import annotations

import torch
from torch import Tensor


def pairwise_distance(xy: Tensor) -> Tensor:
    """xy [B, N, 2] -> [B, N, N] Euclidean distances."""
    delta = xy.unsqueeze(2) - xy.unsqueeze(1)
    return torch.linalg.vector_norm(delta, dim=-1)
