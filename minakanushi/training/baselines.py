"""Non-neural baselines. Beating random is not a result."""

from __future__ import annotations

from torch import Tensor


def constant_position(xy: Tensor, horizon: int) -> Tensor:
    """[B, N, 2] -> [B, H, N, 2] freeze current position."""
    return xy.unsqueeze(1).expand(-1, horizon, -1, -1).clone()


def constant_velocity(xy: Tensor, vel: Tensor, dt: float, horizon: int) -> Tensor:
    """Linear kinematic rollout without learned residual."""
    frames = []
    cur = xy
    for k in range(1, horizon + 1):
        cur = xy + vel * (dt * k)
        frames.append(cur)
    return torch_stack(frames)


def torch_stack(frames: list[Tensor]) -> Tensor:
    import torch

    return torch.stack(frames, dim=1)


def no_memory_state(xy: Tensor, observed_mask: Tensor) -> Tensor:
    """Drop unobserved slots — the anti-persistence estimator."""
    return xy * observed_mask.to(xy.dtype).unsqueeze(-1)


def single_future(xy: Tensor, vel: Tensor, dt: float, horizon: int) -> Tensor:
    """One constant-velocity branch, no strategy-conditioned split."""
    return constant_velocity(xy, vel, dt, horizon)
