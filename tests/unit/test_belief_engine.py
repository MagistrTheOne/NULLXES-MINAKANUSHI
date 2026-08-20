"""Gate 05 Belief Engine: mean + std + existence, not a point kinematic bag."""

from __future__ import annotations

import torch

from helpers import ROOT
from minakanushi.architecture.config import TrainingConfig, load_architecture
from minakanushi.architecture.mina_unit import MinaUnit, pack_units
from minakanushi.architecture.outputs import PositionState
from minakanushi.core.dynamic_world_core import DynamicWorldCore
from minakanushi.state.constructor import StateConstructor, empty_world_state
from minakanushi.training.objectives import belief_nll, compute_objectives, existence_bce
from minakanushi.uncertainty.engine import UncertaintyEngine


def _unit(config, x: float, y: float, vx: float, t: float, eid: int = 11, conf: float = 0.95) -> MinaUnit:
    dim = config.latent_dim
    return MinaUnit(
        source_type="vector",
        source_id=2,
        timestamp=t,
        sequence_index=0,
        spatial_frame="arena",
        spatial_position=(x, y, 0.0),
        spatial_valid=True,
        semantic_embedding=torch.zeros(dim),
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


def _zeros_position(batch: int, n_obs: int, dim: int, device, dtype) -> PositionState:
    z = torch.zeros(batch, n_obs, dim, device=device, dtype=dtype)
    return PositionState(
        embedding=z,
        temporal_embedding=z,
        spatial_embedding=z,
        episode_embedding=z,
        memory_embedding=z,
        source_embedding=z,
        sequence_embedding=z,
    )


def test_as_belief_aliases_kinematics() -> None:
    config = load_architecture(ROOT / "configs" / "architecture" / "cpu_dev.yaml")
    world = empty_world_state(config, 1, device=torch.device("cpu"), dtype=torch.float32)
    view = world.as_belief()
    assert view.position_mean.data_ptr() == world.entity_xy.data_ptr()
    assert view.position_std.data_ptr() == world.xy_std.data_ptr()
    assert view.existence_probability.data_ptr() == world.existence.data_ptr()
    assert float(world.existence[0, 0]) == 1.0


def test_occlusion_inflates_std_and_keeps_existence() -> None:
    config = load_architecture(ROOT / "configs" / "architecture" / "cpu_dev.yaml")
    ctor = StateConstructor(config)
    u_engine = UncertaintyEngine(config)
    device = torch.device("cpu")
    dtype = torch.float32
    world = empty_world_state(config, 1, device=device, dtype=dtype)
    seen = _pack(config, [_unit(config, 2.0, 2.0, 0.4, 0.0)], 0.0, 0.0, device, dtype)
    world = ctor.apply(seen, world, seen.semantic_embedding)
    slot = int((world.entity_id == 11).nonzero(as_tuple=False)[0, 1].item())
    std0 = float(world.xy_std[0, slot].mean())
    exist0 = float(world.existence[0, slot])
    u0 = float(u_engine(world, seen).channels[0, slot, 0])
    empty = _pack(config, [], 0.1, 1.0, device, dtype)
    world = ctor.apply(empty, world, empty.semantic_embedding)
    std1 = float(world.xy_std[0, slot].mean())
    exist1 = float(world.existence[0, slot])
    u1 = float(u_engine(world, empty).channels[0, slot, 0])
    assert std1 > std0
    assert u1 > u0
    assert exist1 > 0.0
    assert exist1 <= exist0
    assert bool(world.occupied[0, slot])


def test_memory_hints_shift_unobserved_mean() -> None:
    config = load_architecture(ROOT / "configs" / "architecture" / "cpu_dev.yaml")
    ctor = StateConstructor(config)
    device = torch.device("cpu")
    dtype = torch.float32
    dim = config.latent_dim
    world = empty_world_state(config, 1, device=device, dtype=dtype)
    seen = _pack(config, [_unit(config, 3.0, 1.0, 0.0, 0.0)], 0.0, 0.0, device, dtype)
    world = ctor.apply(seen, world, seen.semantic_embedding)
    slot = int((world.entity_id == 11).nonzero(as_tuple=False)[0, 1].item())
    empty = _pack(config, [], 0.1, 1.0, device, dtype)
    hints = torch.zeros(1, config.world_slots, dim, device=device, dtype=dtype)
    hints[0, slot, 0] = 4.0
    hints[0, slot, 1] = -2.0
    with_hints = ctor.apply(empty, world, empty.semantic_embedding, memory_hints=hints)
    without = ctor.apply(empty, world, empty.semantic_embedding, memory_hints=None)
    assert not torch.allclose(with_hints.entity_xy[0, slot], without.entity_xy[0, slot])
    assert torch.allclose(with_hints.entity_xy[0, slot], without.entity_xy[0, slot], atol=1.0)


def test_dwc_residual_affects_unobserved_belief() -> None:
    config = load_architecture(ROOT / "configs" / "architecture" / "cpu_dev.yaml")
    ctor = StateConstructor(config)
    device = torch.device("cpu")
    dtype = torch.float32
    world = empty_world_state(config, 1, device=device, dtype=dtype)
    seen = _pack(config, [_unit(config, 2.5, 1.0, 0.5, 0.0)], 0.0, 0.0, device, dtype)
    world = ctor.apply(seen, world, seen.semantic_embedding)
    empty = _pack(config, [], 0.1, 1.0, device, dtype)
    world = ctor.apply(empty, world, empty.semantic_embedding)
    slot = int((world.entity_id == 11).nonzero(as_tuple=False)[0, 1].item())
    assert float(world.age_unobserved[0, slot]) >= 1.0
    torch.manual_seed(0)
    core = DynamicWorldCore(config)
    n_obs = empty.semantic_embedding.shape[1]
    pos = _zeros_position(1, n_obs, config.latent_dim, device, dtype)
    mem = torch.zeros(1, config.world_slots, config.latent_dim, device=device, dtype=dtype)
    on = core(
        world,
        empty.semantic_embedding,
        mem,
        pos,
        empty,
        cognition_budget=1,
    ).world_state
    with torch.no_grad():
        core.xy_residual.weight.zero_()
        core.xy_residual.bias.zero_()
    off = core(
        world,
        empty.semantic_embedding,
        mem,
        pos,
        empty,
        cognition_budget=1,
    ).world_state
    mean_changed = not torch.allclose(on.entity_xy[0, slot], off.entity_xy[0, slot])
    std_changed = not torch.allclose(on.xy_std[0, slot], off.xy_std[0, slot])
    assert mean_changed or std_changed


def test_l_belief_nll_and_existence() -> None:
    mean = torch.zeros(1, 2, 2)
    true = torch.zeros(1, 2, 2)
    true[0, 0] = 1.0
    mask = torch.ones(1, 2, dtype=torch.bool)
    tight = belief_nll(mean, torch.ones(1, 2, 2) * 0.01, true, mask)
    wide = belief_nll(mean, torch.ones(1, 2, 2) * 1.0, true, mask)
    assert float(tight) > float(wide)
    present = existence_bce(torch.ones(1, 2), torch.ones(1, 2), mask)
    absent = existence_bce(torch.ones(1, 2), torch.zeros(1, 2), mask)
    assert float(absent) > float(present)
    train = TrainingConfig()
    dummy = torch.zeros(1, 2, 2)
    occ = torch.ones(1, 2, dtype=torch.bool)
    future = dummy.unsqueeze(1).expand(1, 2, 2, 2).contiguous()
    unc = torch.ones(1, 2, 8) * 0.2
    breakdown = compute_objectives(
        pred_xy=mean,
        true_xy=true,
        occupied=occ,
        pred_next_xy=mean,
        true_next_xy=true,
        pred_future_xy=future,
        true_future_xy=future,
        uncertainty=unc,
        memory_xy=mean,
        memory_true_xy=true,
        memory_mask=occ,
        causal_pred=mean,
        causal_true=mean,
        alt_future_xy=future,
        intra_branch_xy=future,
        latent=torch.zeros(1, 2, 8),
        training=train,
        xy_std=torch.ones(1, 2, 2) * 0.2,
        existence=torch.ones(1, 2),
        true_present=occ,
        hypothesized=occ,
    )
    assert "belief" in breakdown.terms
    assert float(breakdown.terms["belief"]) >= 0.0
