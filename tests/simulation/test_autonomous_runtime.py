"""Gate 09 Autonomous Runtime: cycle(), modes, restore. Not personality."""

from __future__ import annotations

import torch

from helpers import cpu_config
from minakanushi.identity.authority import AuthorityMode
from minakanushi.policy.intent import ActionIntent
from minakanushi.runtime.loop import MinakanushiRuntime
from minakanushi.runtime.state import RuntimeState


def _runtime(seed: int = 9) -> MinakanushiRuntime:
    return MinakanushiRuntime(cpu_config(), seed=seed)


def test_09_runtime_state_is_not_self_model() -> None:
    rt = _runtime()
    assert isinstance(rt.state, RuntimeState)
    assert rt.session.self_model is not None
    assert rt.state.cycle_id == 0
    assert rt.session.self_model.instance.instance_id
    assert rt.state.mode == rt.session.authority.mode.value


def test_09_continuous_cycle_keeps_state() -> None:
    rt = _runtime(3)
    a = rt.cycle()
    b = rt.cycle()
    c = rt.cycle()
    assert [a.runtime.cycle_id, b.runtime.cycle_id, c.runtime.cycle_id] == [1, 2, 3]
    assert c.runtime.runtime_time > a.runtime.runtime_time
    assert c.session.world.occupied.any()
    assert c.telemetry.future_branches > 0
    assert rt.metrics.runtime_cycles == 3
    assert rt.metrics.belief_updates == 3
    assert rt.metrics.prediction_updates == 3
    assert rt.metrics.action_attempts == 3


def test_09_no_operator_still_updates_cognition() -> None:
    rt = _runtime(4)
    rt.set_mode(AuthorityMode.MANUAL)
    first = rt.cycle()
    second = rt.cycle()
    assert first.action_intent.objective == "SAFE_HOLD"
    assert first.telemetry.future_branches > 0
    assert first.telemetry.entity_count >= 1
    assert first.session.focus is not None
    assert first.session.world.occupied.any()
    assert second.action_intent.objective == "SAFE_HOLD"
    assert second.session.self_model is not None
    assert len(second.session.self_model.experience.records) >= 1
    assert second.telemetry.extras["authority_block"] is True


def test_09_authority_modes_same_seed() -> None:
    go = ActionIntent("return", "RETURN", tuple(cpu_config().simulation.home), {}, 1.0, 1e9, (), "operator")
    expected_hold = {
        AuthorityMode.ADVISORY: True,
        AuthorityMode.MANUAL: True,
        AuthorityMode.SAFE_HOLD: True,
        AuthorityMode.DIRECTED: False,
        AuthorityMode.AUTONOMOUS: False,
    }
    for mode, hold in expected_hold.items():
        rt = _runtime(7)
        rt.set_mode(mode)
        result = rt.cycle(operator_intent=go if mode == AuthorityMode.DIRECTED else None)
        assert result.telemetry.future_branches > 0, mode
        assert result.telemetry.extras["authority_mode"] == mode.value
        if hold:
            assert result.action_intent.objective == "SAFE_HOLD", mode
            assert result.metrics.authority_blocks >= 1, mode
        if mode == AuthorityMode.ADVISORY:
            assert result.action_intent.objective == "SAFE_HOLD"
            assert result.telemetry.extras["strategy_proposal"]
            assert result.telemetry.extras["strategy_proposal"] != ""
        if mode == AuthorityMode.AUTONOMOUS:
            assert result.telemetry.extras["authority_block"] is False


def test_09_policy_off_does_not_bypass_constraints() -> None:
    rt = _runtime(6)
    rt.set_mode(AuthorityMode.AUTONOMOUS, policy_enabled=False)
    raid = ActionIntent("raid_restricted", "MOVE_TO", (8.2, 8.2), {}, 1.0, 1e9, (), "operator")
    result = rt.cycle(operator_intent=raid)
    assert result.action_intent.objective == "SAFE_HOLD"
    assert result.action_intent.strategy_id != "raid_restricted"
    assert result.telemetry.future_branches > 0
    assert result.telemetry.extras["authority_block"] is True


def test_09_checkpoint_restore_continues_cycle(tmp_path) -> None:
    live = _runtime(11)
    live.run(5)
    path = tmp_path / "runtime.mina"
    live.save_checkpoint(path)
    before_id = live.session.self_model.instance.instance_id
    before_cycle = live.state.cycle_id
    before_xy = live.session.world.entity_xy.detach().cpu().clone()
    before_exist = live.session.world.existence.detach().cpu().clone()
    before_focus = live.session.focus.to_dict() if live.session.focus else {}
    before_auth = live.session.authority.to_dict()
    before_exp = len(live.session.self_model.experience.records)
    before_mem = live.engine.system.memory.episodic.embeddings.detach().cpu().clone()
    before_write = int(live.engine.system.memory.episodic.write_index)
    before_runtime = live.state.to_dict()
    live.stop()

    restored = _runtime(99)
    restored.restore_checkpoint(path)
    assert restored.state.cycle_id == before_cycle
    assert restored.session.self_model.instance.instance_id == before_id
    assert torch.allclose(restored.session.world.entity_xy.cpu(), before_xy)
    assert torch.allclose(restored.session.world.existence.cpu(), before_exist)
    assert restored.session.focus.to_dict() == before_focus
    assert restored.session.authority.to_dict() == before_auth
    assert len(restored.session.self_model.experience.records) == before_exp
    assert torch.allclose(restored.engine.system.memory.episodic.embeddings.cpu(), before_mem)
    assert int(restored.engine.system.memory.episodic.write_index) == before_write
    assert restored.state.mode == before_runtime["mode"]
    assert restored.metrics.checkpoint_restores >= 1
    assert restored.active is True

    nxt = restored.cycle()
    assert nxt.runtime.cycle_id == before_cycle + 1
    assert len(nxt.session.self_model.experience.records) >= before_exp
    assert nxt.metrics.runtime_cycles == live.metrics.runtime_cycles + 1


def test_09_metrics_keys_present() -> None:
    rt = _runtime(2)
    rt.cycle()
    keys = set(rt.metrics.to_dict())
    assert keys == {
        "runtime_cycles",
        "belief_updates",
        "memory_writes",
        "focus_changes",
        "prediction_updates",
        "action_attempts",
        "authority_blocks",
        "checkpoint_restores",
        "experience_records",
    }
