"""Uncertainty calibration target: predicted uncertainty vs residual error."""

from __future__ import annotations

import torch
from torch import Tensor


def gaussian_nll(error: Tensor, uncertainty: Tensor) -> Tensor:
    """error [..., D], uncertainty [...] or [..., U].

    Interprets mean uncertainty channel as std. Lower-bounded to avoid log(0).
    """
    sigma = uncertainty.mean(dim=-1) if uncertainty.ndim == error.ndim else uncertainty
    sigma = sigma.clamp_min(1e-3)
    sq = error.pow(2).mean(dim=-1)
    return 0.5 * (sq / sigma.pow(2) + 2.0 * torch.log(sigma))
