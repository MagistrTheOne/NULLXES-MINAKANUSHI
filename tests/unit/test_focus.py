"""Gate 07 Focus Engine: attention value, not desire. Not an action generator."""

from __future__ import annotations

import torch

from helpers import ROOT
from minakanushi.architecture.config import load_architecture
from minakanushi.architecture.mina_unit import MinaUnit, pack_units
from minakanushi.focus.engine import FocusEngine, FocusState, FocusType
from minakanushi.identity.experience import ExperienceLog
from minakanushi.memory.experience import ExperienceEngine
from minakanushi.situation.core import SituationCore
from minakanushi.state.constructor import StateConstructor, empty_world_state
from minakanushi.strategy.engine import StrategyEngine
from minakanushi.uncertainty.engine import UncertaintyEngine


def _unit(config, x: float, y: float, vx: float, t: float, eid: int, conf: float = 0.95) -> MinaUnit:
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


def test_uncertainty_attraction_prefers_unknown() -> None:
    config = load_architecture(ROOT / "configs" / "architecture" / "cpu_dev.yaml")
    ctor = StateConstructor(config)
    device = torch.device("cpu")
    dtype = torch.float32
    known = _unit(config, 1.0, 1.0, 0.0, 0.0, eid=11, conf=0.95)
    unknown = _unit(config, 4.0, 1.0, 0.0, 0.0, eid=12, conf=0.55)
    packed = _pack(config, [known, unknown], 0.0, 0.0, device, dtype)
    world = ctor.apply(packed, empty_world_state(config, 1, device=device, dtype=dtype), packed.semantic_embedding)
    focus = FocusEngine().select(world, ExperienceLog(), now=0.0)
    assert focus.target_id == 12
    assert focus.focus_type in {FocusType.NOVELTY.value, FocusType.UNCERTAINTY_REDUCTION.value}
    assert focus.focus_type != FocusType.MAINTENANCE.value


def test_prediction_failure_attracts_revision() -> None:
    config = load_architecture(ROOT / "configs" / "architecture" / "cpu_dev.yaml")
    ctor = StateConstructor(config)
    exp = ExperienceEngine()
    device = torch.device("cpu")
    dtype = torch.float32
    moving = _pack(config, [_unit(config, 2.5, 1.0, -0.8, 0.0, eid=11)], 0.0, 0.0, device, dtype)
    stopped = _pack(config, [_unit(config, 2.5, 1.0, 0.0, 0.1, eid=11)], 0.1, 1.0, device, dtype)
    prior = ctor.apply(moving, empty_world_state(config, 1, device=device, dtype=dtype), moving.semantic_embedding)
    nxt = ctor.apply(stopped, prior, stopped.semantic_embedding)
    log = ExperienceLog()
    for rec in exp.record_cycle(prior, nxt, config.dt, "OBSERVE", 0.1):
        log.append(rec)
    focus = FocusEngine().select(nxt, log, now=0.1)
    assert focus.target_id == 11
    assert focus.focus_type == FocusType.PREDICTION_ERROR.value


def test_stale_memory_loses_to_new_evidence() -> None:
    config = load_architecture(ROOT / "configs" / "architecture" / "cpu_dev.yaml")
    ctor = StateConstructor(config)
    device = torch.device("cpu")
    dtype = torch.float32
    first = _pack(config, [_unit(config, 1.0, 5.0, 0.0, 0.0, eid=11, conf=0.8)], 0.0, 0.0, device, dtype)
    world = ctor.apply(first, empty_world_state(config, 1, device=device, dtype=dtype), first.semantic_embedding)
    empty = _pack(config, [], 0.1, 1.0, device, dtype)
    for _ in range(5):
        world = ctor.apply(empty, world, empty.semantic_embedding)
    sharp = _pack(config, [_unit(config, 4.0, 5.0, 0.0, 2.0, eid=11, conf=0.95)], 2.0, 6.0, device, dtype)
    world = ctor.apply(sharp, world, sharp.semantic_embedding)
    slot = int((world.entity_id == 11).nonzero(as_tuple=False)[0, 1].item())
    assert float(world.uncertainty[0, slot, 2]) > 0.0
    assert float(world.age_unobserved[0, slot]) == 0.0
    focus = FocusEngine().select(world, ExperienceLog(), now=2.0)
    assert focus.focus_type != FocusType.MEMORY_CONFLICT.value


def test_no_hallucinated_goals_when_stable() -> None:
    config = load_architecture(ROOT / "configs" / "architecture" / "cpu_dev.yaml")
    ctor = StateConstructor(config)
    device = torch.device("cpu")
    dtype = torch.float32
    a = _unit(config, 1.0, 1.0, 0.0, 0.0, eid=11, conf=0.97)
    b = _unit(config, 3.0, 1.0, 0.0, 0.0, eid=12, conf=0.97)
    packed = _pack(config, [a, b], 0.0, 0.0, device, dtype)
    world = ctor.apply(packed, empty_world_state(config, 1, device=device, dtype=dtype), packed.semantic_embedding)
    focus = FocusEngine().select(world, ExperienceLog(), now=0.0)
    assert focus.focus_type == FocusType.MAINTENANCE.value
    assert focus.target_id == 0
    u_engine = UncertaintyEngine(config)
    packed_empty_obs = packed
    sit_keep = SituationCore().build(world, u_engine(world, packed_empty_obs), (), focus=focus)
    sit_forced = SituationCore().build(
        world,
        u_engine(world, packed_empty_obs),
        (),
        focus=FocusState(target_id=11, focus_type=FocusType.NOVELTY.value, priority=1.0),
    )
    home = (0.0, 0.0)
    ids_keep = {c.strategy_id for c in StrategyEngine().generate(sit_keep, home)}
    ids_forced = {c.strategy_id for c in StrategyEngine().generate(sit_forced, home)}
    assert ids_keep == ids_forced
    assert all("adventure" not in g.lower() for g in sit_keep.goals)
