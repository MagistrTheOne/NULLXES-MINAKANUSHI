"""Closed loop: ActionIntent changes the subsequent observation."""

from __future__ import annotations

from helpers import build_engine, cpu_config
from minakanushi.policy.intent import ActionIntent
from simulations.synthetic_world.world import SyntheticWorld


def test_intent_changes_next_observation() -> None:
    engine = build_engine()
    world = SyntheticWorld(engine.config.simulation, seed=2)
    state = engine.initialize()
    obs0 = world.observe()
    start = obs0.agent_xy
    result = engine.step(obs0, state)
    world.step(result.action_intent)
    obs1 = world.observe()
    moved = (obs1.agent_xy[0] - start[0]) ** 2 + (obs1.agent_xy[1] - start[1]) ** 2
    held = result.action_intent.objective in {"WAIT", "OBSERVE", "SAFE_HOLD", "ABORT", "REQUEST_ASSISTANCE"}
    if held:
        assert moved < 1e-6
    else:
        assert moved > 0.0
    assert obs1.timestamp > obs0.timestamp


def test_unused_observe_after_step_advances_sensor_rng() -> None:
    """One observe per cycle. A discarded observe() draws noise and skips the next sample."""
    cfg = cpu_config().simulation
    wait = ActionIntent("wait", "WAIT", (1.0, 1.0), {}, 1.0, 1e9, (), "test")
    honest = SyntheticWorld(cfg, seed=2)
    burned = SyntheticWorld(cfg, seed=2)

    first_honest = honest.observe()
    first_burned = burned.observe()
    assert first_honest.visible == first_burned.visible
    assert len(first_honest.visible) > 0

    honest.step(wait)
    burned.step(wait)
    burned.observe()
    next_honest = honest.observe()
    next_burned = burned.observe()
    assert next_honest.visible != next_burned.visible

