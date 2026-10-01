"""S2: perception drops corrupt vector items, fails loud on bad telemetry."""

from __future__ import annotations

import pytest
import torch

from minakanushi.perception.bridge import Observation, PerceptionBridge
from helpers import cpu_config


def _obs(**kw):
    base = dict(
        timestamp=0.0, agent_xy=(1.0, 1.0), agent_vel=(0.0, 0.0),
        heading=0.0, health=1.0, battery=1.0,
    )
    base.update(kw)
    return Observation(**base)


def _good_item(iid: int = 11, conf: float = 0.9):
    return {"id": iid, "kind": "mover", "xy": (2.0, 2.0), "vel": (0.1, 0.0), "confidence": conf}


def test_corrupt_items_dropped_cycle_survives() -> None:
    cfg = cpu_config().architecture
    bridge = PerceptionBridge(cfg)
    device, dtype = torch.device("cpu"), torch.float32
    obs = _obs(visible=(
        _good_item(11),
        {"id": 0, "kind": "mover", "xy": (2.0, 2.0), "confidence": 0.9},
        {"id": 12, "kind": "mover", "xy": (float("nan"), 2.0), "confidence": 0.9},
        {"id": 13, "kind": "mover", "xy": (2.0, 2.0), "confidence": 2.0},
        {"id": 14, "kind": "mover", "xy": (2.0, float("inf")), "confidence": 0.9},
        {"id": 15, "kind": "mover", "confidence": 0.9},
    ))
    units = bridge.encode(obs, device=device, dtype=dtype)
    eids = sorted(u.entity_reference for u in units)
    assert eids == [1, 11]
    assert units[0].metadata["dropped_items"] == 5


def test_bad_telemetry_raises() -> None:
    cfg = cpu_config().architecture
    bridge = PerceptionBridge(cfg)
    with pytest.raises(ValueError, match="non-finite"):
        bridge.encode(_obs(agent_xy=(float("nan"), 1.0)), device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(ValueError, match="noise_std"):
        bridge.encode(_obs(noise_std=-1.0), device=torch.device("cpu"), dtype=torch.float32)
