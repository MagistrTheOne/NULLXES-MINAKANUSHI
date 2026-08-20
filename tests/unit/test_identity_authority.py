"""Gate 04: SelfModel, Authority, Persona — structured identity, not a prompt."""

from __future__ import annotations

from helpers import build_engine, cpu_config
from minakanushi.identity.authority import AuthorityMode
from minakanushi.identity.constants import ARCHITECTURE_ID, ARCHITECTURE_NAME, ORGANIZATION, SHORT_NAME
from minakanushi.identity.persona import PersonaModel
from minakanushi.identity.self_model import SelfModel
from minakanushi.policy.intent import ActionIntent
from simulations.synthetic_world.world import SyntheticWorld


def test_self_model_is_not_a_world_entity() -> None:
    engine = build_engine()
    state = engine.initialize()
    assert state.self_model is not None
    assert state.self_model.is_world_entity() is False
    assert state.self_model.identity.architecture_name == ARCHITECTURE_NAME
    assert state.self_model.identity.short_name == SHORT_NAME
    assert state.self_model.identity.architecture_id == ARCHITECTURE_ID
    assert state.self_model.identity.organization == ORGANIZATION
    ids = state.world.entity_id[0, state.world.occupied[0]].tolist()
    assert ARCHITECTURE_NAME not in ids
    assert SHORT_NAME not in [str(x) for x in ids]


def test_identity_exists_without_prompt() -> None:
    model = SelfModel.from_config(cpu_config().architecture.identity, cpu_config().architecture)
    dumped = model.to_dict()
    assert "you are" not in str(dumped).lower()
    assert dumped["identity"]["architecture_name"] == "MINAKANUSHI"
    assert dumped["identity"]["short_name"] == "MINA"


def test_persona_does_not_enter_cognition() -> None:
    e1 = build_engine()
    e1.persona = PersonaModel(communication_style="precise_operational")
    e2 = build_engine()
    e2.persona = PersonaModel(communication_style="verbose_narrative")
    w1 = SyntheticWorld(e1.config.simulation, seed=4)
    w2 = SyntheticWorld(e2.config.simulation, seed=4)
    r1 = e1.step(w1.observe(), e1.initialize())
    r2 = e2.step(w2.observe(), e2.initialize())
    assert float((r1.state.world.entity_xy - r2.state.world.entity_xy).abs().sum()) == 0.0
    assert r1.action_intent.objective == r2.action_intent.objective
    assert e2.persona.affects_cognition() is False
    assert e2.persona.feminine_presenting is True


def test_policy_off_does_not_disable_cognition() -> None:
    engine = build_engine()
    engine.set_mode(AuthorityMode.AUTONOMOUS, policy_enabled=False)
    world = SyntheticWorld(engine.config.simulation, seed=5)
    state = engine.initialize()
    state.authority = engine.authority
    obs = world.observe()
    result = engine.step(obs, state)
    assert result.telemetry.future_branches > 0
    assert result.telemetry.entity_count >= 1
    assert result.telemetry.memory_writes >= 0
    assert result.state.world.occupied.any()
    assert result.action_intent.objective == "SAFE_HOLD"
    assert "policy_off" in result.action_intent.provenance or "authority" in result.action_intent.provenance


def test_policy_off_cannot_emit_unrestricted_move() -> None:
    engine = build_engine()
    engine.set_mode(AuthorityMode.AUTONOMOUS, policy_enabled=False)
    world = SyntheticWorld(engine.config.simulation, seed=6)
    state = engine.initialize()
    state.authority = engine.authority
    raid = ActionIntent("raid_restricted", "MOVE_TO", (8.2, 8.2), {}, 1.0, 1e9, (), "operator")
    result = engine.step(world.observe(), state, operator_intent=raid)
    assert result.action_intent.objective == "SAFE_HOLD"
    assert result.action_intent.strategy_id != "raid_restricted"


def test_authority_modes_expected_intents() -> None:
    engine = build_engine()
    world = SyntheticWorld(engine.config.simulation, seed=7)
    go = ActionIntent("return", "RETURN", tuple(engine.config.simulation.home), {}, 1.0, 1e9, (), "operator")
    for mode, expect_hold in (
        (AuthorityMode.ADVISORY, True),
        (AuthorityMode.MANUAL, True),
        (AuthorityMode.SAFE_HOLD, True),
        (AuthorityMode.DIRECTED, False),
        (AuthorityMode.AUTONOMOUS, False),
    ):
        engine.set_mode(mode)
        state = engine.initialize()
        state.authority = engine.authority
        result = engine.step(world.observe(), state, operator_intent=go if mode == AuthorityMode.DIRECTED else None)
        if expect_hold:
            assert result.action_intent.objective == "SAFE_HOLD", mode
        else:
            assert result.telemetry.future_branches > 0, mode
        assert result.telemetry.extras["authority_mode"] == mode.value


def test_self_model_survives_mina_roundtrip(tmp_path) -> None:
    engine = build_engine()
    world = SyntheticWorld(engine.config.simulation, seed=8)
    state = engine.initialize()
    state = engine.step(world.observe(), state).state
    state.self_model.instance.history_reference = "gate04"
    path = tmp_path / "identity.mina"
    engine.save_checkpoint(path, state)
    fresh = build_engine()
    loaded = fresh.initialize()
    loaded = fresh.load_checkpoint(path, loaded)
    assert loaded.self_model is not None
    assert loaded.self_model.identity.short_name == "MINA"
    assert loaded.self_model.identity.architecture_name == "MINAKANUSHI"
    assert loaded.self_model.instance.history_reference == "gate04"
    assert loaded.authority is not None
    assert loaded.persona is not None
    assert loaded.persona.full_name == "MINAKANUSHI"
