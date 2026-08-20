"""Uncertainty calibration target: predicted uncertainty vs residual error."""

from __future__ import annotations

import torch
from torch import Tensor


def gaussian_nll(error: Tensor, uncertainty: Tensor, channel: int | None = None) -> Tensor:
    """error [..., D], uncertainty [..., U] or [...].

    Channel 6 (state) is the position-error sigma. Mean-of-all-channels is forbidden:
    missing/conflict/noise are not interchangeable with state uncertainty.
    """
    if uncertainty.shape[-1] == error.shape[-1]:
        sigma = uncertainty.mean(dim=-1)
    elif channel is None:
        from minakanushi.state.correction import STATE_CHANNEL

        sigma = uncertainty[..., STATE_CHANNEL]
    else:
        sigma = uncertainty[..., channel]
    sigma = sigma.clamp_min(1e-3)
    sq = error.pow(2).mean(dim=-1)
    return 0.5 * (sq / sigma.pow(2) + 2.0 * torch.log(sigma))
