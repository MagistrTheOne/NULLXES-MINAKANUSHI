"""Convergence of internal cognition cycles."""

from __future__ import annotations

import torch
from torch import Tensor


def slot_delta(previous: Tensor, current: Tensor, occupied: Tensor) -> Tensor:
    """Mean occupied-slot L2 change. shape out: [B]"""
    delta = torch.linalg.vector_norm(current - previous, dim=-1)
    denom = occupied.to(delta.dtype).sum(dim=-1).clamp_min(1.0)
    return (delta * occupied.to(delta.dtype)).sum(dim=-1) / denom
