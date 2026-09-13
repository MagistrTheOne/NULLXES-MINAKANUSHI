"""YMBOT-C laboratory tests. Run from lab/ymbot_c: python -m pytest tests -q"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mina_loop import (  # noqa: E402
    IDENTITY,
    ActionIntent,
    LabConfig,
    MinakanushiLabEngine,
    StrategyCandidate,
    SyntheticWorld,
    future_separation,
    run_closed_loop,
    run_selftest,
    train_world_residual,
)


def test_selftest_passes() -> None:
    report = run_selftest()
    assert report["passed"], report


def test_identity_is_metadata_not_prompt() -> None:
    assert IDENTITY["architecture"] == "MINAKANUSHI"
    assert IDENTITY["organization"] == "NULLXES"
    assert IDENTITY["pwm"] is False
    assert IDENTITY["accepted"] is False


def test_intent_has_no_motor_pwm() -> None:
    loop = run_closed_loop(steps=8, seed=11)
    assert loop["identity"]["pwm"] is False
    intent = ActionIntent("safe_hold", "SAFE_HOLD", (1.0, 1.0), {}, 1.0, 1.0, (), "t")
    dumped = intent.to_dict()
    assert dumped["pwm"] is False
    assert "duty" not in dumped
    assert "motor" not in dumped


def test_hidden_entity_persists() -> None:
    cfg = LabConfig(seed=11)
    world = SyntheticWorld(cfg)
    engine = MinakanushiLabEngine(cfg)
    engine.step(world.observe())
    world.hidden_ids.add(11)
    for _ in range(3):
        world.step(ActionIntent("wait", "WAIT", tuple(world.agent.xy), {}, 1.0, 1e9, (), "t"))
        out = engine.step(world.observe())
    assert 11 in out.belief.entities
    assert out.belief.entities[11].uncertainty > 0.1


def test_hard_zone_beats_high_value() -> None:
    engine = MinakanushiLabEngine(LabConfig(seed=11))
    engine.step(SyntheticWorld(LabConfig(seed=11)).observe())
    raid = StrategyCandidate("raid", "MOVE_TO", (8.2, 8.2), 99.0, 0.0, {"speed": 1.0})
    hold = StrategyCandidate("safe_hold", "SAFE_HOLD", (1.0, 1.0), -20.0, 0.1)
    futures = {c.strategy_id: engine._futures(c) for c in (raid, hold)}
    allowed, rejected, _ = engine.kernel.filter([raid, hold], futures)
    assert any(c.strategy_id == "raid" for c in rejected)
    assert all(a.strategy_id != "raid" for a in allowed)
    chosen = engine._select(allowed, 0.0)
    assert chosen.strategy_id != "raid"


def test_wait_and_move_futures_differ() -> None:
    engine = MinakanushiLabEngine(LabConfig(seed=11))
    engine.step(SyntheticWorld(LabConfig(seed=11)).observe())
    assert future_separation(engine) > 0.15


def test_a_then_b_is_not_b_then_a() -> None:
    report = run_selftest()
    assert report["results"]["temporal_order"]


def test_policy_off_does_not_erase_world() -> None:
    cfg = LabConfig(seed=11)
    engine = MinakanushiLabEngine(cfg, policy_enabled=False)
    out = engine.step(SyntheticWorld(cfg).observe())
    assert out.action_intent.objective == "SAFE_HOLD"
    assert out.telemetry.policy_enabled is False
    assert out.belief.entities


def test_l1_residual_beats_zero_on_cpu() -> None:
    report = train_world_residual(steps=30, device="cpu", seed=11)
    assert report["constructs_6_8b"] is False
    assert report["ade_trained"] < report["ade_zero"]
    assert report["params"] < 10_000
    assert np.isfinite(report["loss"])


def test_selftest_json_roundtrip() -> None:
    raw = json.dumps(run_selftest())
    parsed = json.loads(raw)
    assert parsed["passed"] is True
