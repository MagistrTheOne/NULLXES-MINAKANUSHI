"""Adversarial reality: change of mind, not pretty prediction."""

from __future__ import annotations

import torch

from helpers import ROOT, cpu_config
from minakanushi.architecture.config import load_architecture
from minakanushi.architecture.mina_unit import MinaUnit, pack_units
from minakanushi.state.constructor import StateConstructor, empty_world_state
from minakanushi.training.metrics import (
    belief_revision_accuracy,
    correction_latency,
    false_persistence_steps,
)
from simulations.synthetic_world.dataset import generate_episode


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


def test_hidden_entity_correction_revises_velocity() -> None:
    config = load_architecture(ROOT / "configs" / "architecture" / "cpu_dev.yaml")
    ctor = StateConstructor(config)
    device = torch.device("cpu")
    dtype = torch.float32
    dim = config.latent_dim
    world = empty_world_state(config, 1, device=device, dtype=dtype)

    def pack(unit, now: float, ep: float):
        return pack_units([unit], batch_index=0, max_units=8, latent_dim=dim, episode_position=ep, now=now, device=device, dtype=dtype)

    seen = pack(_unit(config, 2.5, 1.0, -0.8, 0.0), 0.0, 0.0)
    world = ctor.apply(seen, world, seen.semantic_embedding)
    slot = (world.entity_id == 11).nonzero(as_tuple=False)[0, 1]
    assert float(world.entity_vel[0, slot, 0]) < -0.3
    u_seen = float(world.uncertainty[0, slot].mean())
    conf_seen = float(world.confidence[0, slot])

    empty = pack_units([], batch_index=0, max_units=8, latent_dim=dim, episode_position=1.0, now=0.1, device=device, dtype=dtype)
    for step in range(5):
        world = ctor.apply(empty, world, empty.semantic_embedding)
    assert bool(world.occupied[0, slot])
    u_hidden = float(world.uncertainty[0, slot].mean())
    conf_hidden = float(world.confidence[0, slot])
    assert u_hidden > u_seen
    assert conf_hidden < conf_seen
    old_xy = world.entity_xy[0, slot].detach().clone()
    old_vel = world.entity_vel[0, slot].detach().clone()

    back = pack(_unit(config, 2.5, 1.0, 0.0, 0.6, conf=0.97), 0.6, 6.0)
    world = ctor.apply(back, world, back.semantic_embedding)
    new_xy = world.entity_xy[0, slot]
    new_vel = world.entity_vel[0, slot]
    acc = float(belief_revision_accuracy(old_xy.unsqueeze(0), new_xy.unsqueeze(0), torch.tensor([2.5, 1.0])).mean())
    assert acc == 1.0 or float(torch.linalg.vector_norm(new_xy - torch.tensor([2.5, 1.0]))) < 0.2
    assert float(torch.linalg.vector_norm(new_vel)) < float(torch.linalg.vector_norm(old_vel))
    assert world.corrections
    assert correction_latency(wrong_at=1, evidence_at=6, corrected_at=6) == 0


def test_gone_forever_false_persistence_bounded() -> None:
    sim = cpu_config().simulation
    episode = generate_episode(sim, seed=7, episode_index=0, length=12, scenario="gone_forever")
    gone_from = 3
    occupied_flags = []
    exist_trace = []
    config = load_architecture(ROOT / "configs" / "architecture" / "cpu_dev.yaml")
    ctor = StateConstructor(config)
    device = torch.device("cpu")
    dtype = torch.float32
    world = empty_world_state(config, 1, device=device, dtype=dtype)
    eid = episode.truth[0].entity_id[1] if len(episode.truth[0].entity_id) > 1 else 11
    for t, obs in enumerate(episode.observations):
        from minakanushi.architecture.mina_unit import pack_units
        from minakanushi.perception.bridge import PerceptionBridge

        bridge = PerceptionBridge(config)
        units = bridge.encode(obs, device=device, dtype=dtype)
        packed = pack_units(
            units,
            batch_index=0,
            max_units=config.max_observations,
            latent_dim=config.latent_dim,
            episode_position=float(t),
            now=obs.timestamp,
            device=device,
            dtype=dtype,
        )
        world = ctor.apply(packed, world, packed.semantic_embedding)
        if t >= gone_from:
            hit = (world.entity_id == eid) & world.occupied
            occupied_flags.append(bool(hit.any()))
            if bool(hit.any()):
                slot = int(hit.nonzero(as_tuple=False)[0, 1].item())
                exist_trace.append(float(world.existence[0, slot]))
            else:
                exist_trace.append(0.0)
    persist = false_persistence_steps(occupied_flags)
    assert persist <= config.persistence.steps + 1
    assert exist_trace
    assert exist_trace[0] > 0.0
    assert exist_trace[-1] == 0.0
    positive = [e for e in exist_trace if e > 0.0]
    assert len(positive) >= 1
    if len(positive) >= 2:
        assert positive[-1] < positive[0]
