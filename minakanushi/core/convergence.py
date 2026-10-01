"""Convergence of internal cognition cycles."""

from __future__ import annotations

import torch
from torch import Tensor


def slot_delta(previous: Tensor, current: Tensor, occupied: Tensor) -> Tensor:
    """Mean occupied-slot L2 change. shape out: [B]"""
    delta = torch.linalg.vector_norm(current - previous, dim=-1)
    denom = occupied.to(delta.dtype).sum(dim=-1).clamp_min(1.0)
    return (delta * occupied.to(delta.dtype)).sum(dim=-1) / denom


def rms_delta(previous: Tensor, current: Tensor, occupied: Tensor) -> Tensor:
    """Per-element RMS change over occupied slots. shape out: [B].

    Unlike slot_delta (L2 sum over features), RMS is dimension-free, so a
    64-dim latent and a 2-dim velocity can share one threshold and weights.
    """
    diff = (current - previous).pow(2).mean(dim=-1)
    occ = occupied.to(diff.dtype)
    mean = (diff * occ).sum(dim=-1) / occ.sum(dim=-1).clamp_min(1.0)
    return mean.clamp_min(0.0).sqrt()


def cognition_delta(
    latent_prev: Tensor,
    latent_cur: Tensor,
    xy_prev: Tensor,
    xy_cur: Tensor,
    vel_prev: Tensor,
    vel_cur: Tensor,
    occupied: Tensor,
    *,
    w_latent: float = 1.0,
    w_xy: float = 1.0,
    w_vel: float = 0.5,
) -> Tensor:
    """Combined latent + kinematic convergence score. shape out: [B].

    A cognition loop that settles its latent while its implied kinematics
    still drift has not converged; all three terms must be small.
    """
    return (
        float(w_latent) * rms_delta(latent_prev, latent_cur, occupied)
        + float(w_xy) * rms_delta(xy_prev, xy_cur, occupied)
        + float(w_vel) * rms_delta(vel_prev, vel_cur, occupied)
    )
