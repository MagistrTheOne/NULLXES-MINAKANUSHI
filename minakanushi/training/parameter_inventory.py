"""Closed-form parameter estimates. Not measured by instantiating the model."""

from __future__ import annotations

from minakanushi.architecture.config import ArchitectureConfig


def _linear(in_f: int, out_f: int) -> int:
    return in_f * out_f + out_f


def _fourier(dim: int, freqs: int) -> int:
    return _linear(2 * freqs + 1, dim)


def estimate_parameters(config: ArchitectureConfig) -> dict[str, int]:
    """Formula inventory. Label: ESTIMATE from source math, not torch.numel()."""
    d = config.latent_dim
    f = config.npf.num_frequencies
    s = config.npf.max_sources
    l = config.core_depth
    m = config.memory_dim
    u = config.uncertainty_channels
    k = config.future_branches

    perception = 2 * (_linear(7, d) + _linear(d, d))
    npf = (
        _fourier(d, f)  # sequence
        + 4 * _fourier(d, f) + _linear(4 * d, d)  # physical time bundle
        + (_linear(3, d) + _linear(d, d))  # spatial
        + _fourier(d, f)  # episode
        + _fourier(d, f)  # memory age
        + s * d  # source embedding
        + _linear(6 * d, d) + _linear(d, d)  # mixer
        + _linear(6 * d, 6)  # gate
    )
    block = (
        4 * d  # two LayerNorms
        + 6 * _linear(d, d)  # qkv obs + qkv rel
        + _linear(d, 2 * d) + _linear(2 * d, d)  # ff
        + _linear(2 * d, d)  # gate
    )
    dwc = (
        l * block
        + _linear(2 * d, d)
        + _linear(d, 2)
        + _linear(d, 2)
        + _linear(d, m)
        + _linear(d, d)
    )
    uncertainty = _linear(d, u)
    memory = _linear(m, d)
    future = (
        k * d
        + _linear(2 * d + 4, d)
        + _linear(d, 2)
        + _linear(d, 1)
        + _linear(d, 1)
    )
    total = perception + npf + dwc + uncertainty + memory + future
    return {
        "perception": perception,
        "npf": npf,
        "cognitive_block": block,
        "dwc": dwc,
        "uncertainty": uncertainty,
        "memory_read": memory,
        "future": future,
        "total_estimate": total,
        "world_slots_are_state_not_params": 0,
        "memory_slots_are_buffers_not_params": 0,
        "horizon_is_unroll_not_params": 0,
    }
