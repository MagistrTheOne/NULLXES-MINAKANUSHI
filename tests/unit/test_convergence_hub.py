"""S5: combined convergence, hub research-scale gate by profile+params."""

from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from minakanushi.core.convergence import cognition_delta, rms_delta, slot_delta
from minakanushi.hub import refuse_if_research_scale


def _occ(n: int = 4):
    return torch.ones(1, n, dtype=torch.bool)


def test_rms_delta_is_dimension_free() -> None:
    prev = torch.zeros(1, 4, 64)
    cur = torch.ones(1, 4, 64)
    assert float(rms_delta(prev, cur, _occ())) == pytest.approx(1.0)
    prev2 = torch.zeros(1, 4, 2)
    cur2 = torch.ones(1, 4, 2)
    assert float(rms_delta(prev2, cur2, _occ())) == pytest.approx(1.0)
    # L2 scales with dim; RMS does not.
    assert float(slot_delta(prev, cur, _occ())) == pytest.approx(8.0)


def test_cognition_delta_sees_kinematic_drift() -> None:
    occ = _occ()
    lat = torch.zeros(1, 4, 8)
    xy0 = torch.zeros(1, 4, 2)
    vel = torch.zeros(1, 4, 2)
    settled = cognition_delta(lat, lat, xy0, xy0, vel, vel, occ)
    assert float(settled) == 0.0
    xy1 = torch.ones(1, 4, 2)
    drift = cognition_delta(lat, lat, xy0, xy1, vel, vel, occ)
    assert float(drift) > 0.0


def test_hub_refuses_frozen_profile_even_narrow_latent() -> None:
    narrow_frozen = SimpleNamespace(latent_dim=8, core_depth=32, world_slots=512, memory_slots=1024)
    with pytest.raises(RuntimeError, match="load_mina"):
        refuse_if_research_scale(narrow_frozen)
    refuse_if_research_scale(SimpleNamespace(latent_dim=8, core_depth=2, world_slots=16, memory_slots=32))


def test_hub_refuses_by_param_count() -> None:
    with pytest.raises(RuntimeError, match="parameters="):
        refuse_if_research_scale(SimpleNamespace(latent_dim=8), total_params=6_799_130_646)
    refuse_if_research_scale(SimpleNamespace(latent_dim=8), total_params=1000)
