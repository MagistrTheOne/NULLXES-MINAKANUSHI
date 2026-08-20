"""Gate 06: memory as experience — prediction, reality, error, lesson. Not RAG."""

from __future__ import annotations

import torch

from helpers import ROOT, build_engine
from minakanushi.architecture.config import load_architecture
from minakanushi.architecture.mina_unit import MinaUnit, pack_units
from minakanushi.identity.experience import LESSON_VELOCITY, ExperienceLog
from minakanushi.memory.experience import ExperienceEngine, VEL_LESSON_BOOST
from minakanushi.state.constructor import StateConstructor, empty_world_state
from simulations.synthetic_world.world import SyntheticWorld


def _unit(config, x: float, y: float, vx: float, t: float, eid: int = 11, conf: float = 0.95) -> MinaUnit:
    return MinaUnit(
        source_type="vector",
        source_id=2,
        timestamp=t,
        sequence_index=0,
        spatial_frame="arena",
        spatial_position=(x, y, 0.0),
        spatial_valid=True,
        semantic_embedding=torch.zeros(config.latent_dim),
        confidence=conf,
        uncertainty=1.0 - conf,
        persistence=1.0,
        entity_reference=eid,
        relation_reference=0,
        kind="mover",
        metadata={"vel": (vx, 0.0)},
    )


def _pack(config, units, now: float, ep: float, device, dtype):
    return pack_units(
        units,
        batch_index=0,
        max_units=config.max_observations,
        latent_dim=config.latent_dim,
        episode_position=ep,
        now=now,
        device=device,
        dtype=dtype,
    )


def test_stop_writes_velocity_lesson() -> None:
    config = load_architecture(ROOT / "configs" / "architecture" / "cpu_dev.yaml")
    ctor = StateConstructor(config)
    engine = ExperienceEngine()
    device = torch.device("cpu")
    dtype = torch.float32
    world = empty_world_state(config, 1, device=device, dtype=dtype)
    moving = _pack(config, [_unit(config, 2.5, 1.0, -0.8, 0.0)], 0.0, 0.0, device, dtype)
    world = ctor.apply(moving, world, moving.semantic_embedding)
    stopped = _pack(config, [_unit(config, 2.5, 1.0, 0.0, 0.1)], 0.1, 1.0, device, dtype)
    nxt = ctor.apply(stopped, world, stopped.semantic_embedding)
    recs = engine.record_cycle(world, nxt, config.dt, "OBSERVE", 0.1)
    movers = [r for r in recs if r.entity_id == 11]
    assert movers
    rec = movers[0]
    assert rec.error_vel >= 0.25
    assert rec.lesson == LESSON_VELOCITY
    assert rec.predicted_vel[0] < -0.3
    assert abs(rec.observed_vel[0]) < 0.2


def test_velocity_lesson_inflates_unobserved_vel_std() -> None:
    config = load_architecture(ROOT / "configs" / "architecture" / "cpu_dev.yaml")
    ctor = StateConstructor(config)
    exp = ExperienceEngine()
    device = torch.device("cpu")
    dtype = torch.float32
    moving = _pack(config, [_unit(config, 2.5, 1.0, -0.8, 0.0)], 0.0, 0.0, device, dtype)
    stopped = _pack(config, [_unit(config, 2.5, 1.0, 0.0, 0.1)], 0.1, 1.0, device, dtype)
    moving_world = ctor.apply(moving, empty_world_state(config, 1, device=device, dtype=dtype), moving.semantic_embedding)
    stopped_world = ctor.apply(stopped, moving_world, stopped.semantic_embedding)
    log = ExperienceLog()
    for rec in exp.record_cycle(moving_world, stopped_world, config.dt, "OBSERVE", 0.1):
        log.append(rec)
    slot = int((stopped_world.entity_id == 11).nonzero(as_tuple=False)[0, 1].item())
    empty = _pack(config, [], 0.2, 2.0, device, dtype)
    boost = exp.std_boost(stopped_world, log)
    with_lesson = ctor.apply(empty, stopped_world, empty.semantic_embedding, experience_boost=boost)
    without = ctor.apply(empty, stopped_world, empty.semantic_embedding, experience_boost=None)
    assert float(with_lesson.vel_std[0, slot].mean()) >= float(without.vel_std[0, slot].mean()) + VEL_LESSON_BOOST * 0.9


def test_experience_does_not_override_live_evidence() -> None:
    config = load_architecture(ROOT / "configs" / "architecture" / "cpu_dev.yaml")
    ctor = StateConstructor(config)
    exp = ExperienceEngine()
    device = torch.device("cpu")
    dtype = torch.float32
    moving = _pack(config, [_unit(config, 2.5, 1.0, -0.8, 0.0)], 0.0, 0.0, device, dtype)
    stopped = _pack(config, [_unit(config, 2.5, 1.0, 0.0, 0.1, conf=0.97)], 0.1, 1.0, device, dtype)
    prior = ctor.apply(moving, empty_world_state(config, 1, device=device, dtype=dtype), moving.semantic_embedding)
    dummy_after = ctor.apply(stopped, prior, stopped.semantic_embedding)
    log = ExperienceLog()
    for rec in exp.record_cycle(prior, dummy_after, config.dt, "OBSERVE", 0.1):
        log.append(rec)
    boost = exp.std_boost(prior, log)
    live = ctor.apply(stopped, prior, stopped.semantic_embedding, experience_boost=boost)
    plain = ctor.apply(stopped, prior, stopped.semantic_embedding, experience_boost=None)
    slot = int((live.entity_id == 11).nonzero(as_tuple=False)[0, 1].item())
    assert torch.allclose(live.entity_vel[0, slot], plain.entity_vel[0, slot])
    assert torch.allclose(live.entity_xy[0, slot], plain.entity_xy[0, slot])
    assert torch.allclose(live.vel_std[0, slot], plain.vel_std[0, slot])


def test_runtime_records_experience() -> None:
    engine = build_engine()
    world = SyntheticWorld(engine.config.simulation, seed=11)
    state = engine.initialize()
    first = engine.step(world.observe(), state)
    world.step(first.action_intent)
    second = engine.step(world.observe(), first.state)
    assert second.state.self_model is not None
    assert second.telemetry.extras["experience_count"] == len(second.state.self_model.experience.records)
    assert second.telemetry.extras["experience_count"] >= 1
