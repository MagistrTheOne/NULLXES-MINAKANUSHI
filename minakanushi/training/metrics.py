"""Architectural metrics. Loss is not a substitute for these."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class MetricBundle:
    world_state_position_error: float
    world_state_velocity_error: float
    future_ADE: float
    future_FDE: float
    entity_persistence_accuracy: float
    reacquisition_accuracy: float
    uncertainty_calibration_error: float
    branch_diversity: float
    branch_coverage: float
    memory_effect_delta: float
    constraint_violation_count: int
    closed_loop_success_rate: float


def masked_mse(pred: Tensor, true: Tensor, mask: Tensor) -> Tensor:
    w = mask.to(pred.dtype).unsqueeze(-1)
    return (pred - true).pow(2).mul(w).sum() / w.sum().clamp_min(1.0)


def displacement_error(pred_traj: Tensor, true_traj: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """pred_traj [B,H,N,2], mask [B,N] -> ADE, FDE scalars."""
    w = mask.to(pred_traj.dtype).unsqueeze(1).unsqueeze(-1)
    err = torch.linalg.vector_norm(pred_traj - true_traj, dim=-1)
    ade = (err * w.squeeze(-1)).sum() / w.sum().clamp_min(1.0)
    fde = (err[:, -1] * mask.to(err.dtype)).sum() / mask.to(err.dtype).sum().clamp_min(1.0)
    return ade, fde


def branch_diversity(branch_xy: Tensor) -> Tensor:
    """branch_xy [K, H, N, 2] pairwise terminal distance mean."""
    terminals = branch_xy[:, -1]
    k = terminals.shape[0]
    if k < 2:
        return terminals.new_zeros(())
    total = terminals.new_zeros(())
    count = 0
    for i in range(k):
        for j in range(i + 1, k):
            total = total + torch.linalg.vector_norm(terminals[i] - terminals[j], dim=-1).mean()
            count += 1
    return total / max(count, 1)


def memory_effect_delta(with_memory: Tensor, without_memory: Tensor, mask: Tensor) -> Tensor:
    """L2 change in occupied latents when retrieval is enabled vs zeros."""
    w = mask.to(with_memory.dtype).unsqueeze(-1)
    return (with_memory - without_memory).pow(2).mul(w).sum() / w.sum().clamp_min(1.0)
