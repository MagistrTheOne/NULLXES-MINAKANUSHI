"""WorldState factory and empty-state constructor."""

from __future__ import annotations

import torch
from torch import Tensor

from minakanushi.architecture.config import ArchitectureConfig
from minakanushi.architecture.mina_unit import KIND_IDS, MinaUnitBatch
from minakanushi.state.correction import (
    CONFLICT_CHANNEL,
    NOISY_CHANNEL,
    fuse,
    revise_slot,
)
from minakanushi.state.entity import AGENT_SLOT
from minakanushi.state.world import (
    BELIEF_EXISTENCE_FLOOR,
    BELIEF_STD_MIN,
    COAST_STD_GAIN,
    EXISTENCE_DECAY,
    MEMORY_MEAN_GAIN,
    PRED_CONF_DECAY,
    WorldState,
)
from minakanushi.utils.tensors import assert_finite, assert_shape


def empty_world_state(
    config: ArchitectureConfig,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    timestamp: float = 0.0,
) -> WorldState:
    n = config.world_slots
    d = config.latent_dim
    u = config.uncertainty_channels
    occupied = torch.zeros(batch_size, n, dtype=torch.bool, device=device)
    occupied[:, AGENT_SLOT] = True
    entity_id = torch.zeros(batch_size, n, dtype=torch.long, device=device)
    entity_id[:, AGENT_SLOT] = 1
    kind = torch.zeros(batch_size, n, dtype=torch.long, device=device)
    kind[:, AGENT_SLOT] = KIND_IDS["agent"]
    uncertainty = torch.ones(batch_size, n, u, device=device, dtype=dtype) * 0.5
    xy_std = torch.ones(batch_size, n, 2, device=device, dtype=dtype) * 0.1
    vel_std = torch.ones(batch_size, n, 2, device=device, dtype=dtype) * 0.1
    existence = torch.zeros(batch_size, n, device=device, dtype=dtype)
    existence[:, AGENT_SLOT] = 1.0
    pred_confidence = torch.zeros(batch_size, n, device=device, dtype=dtype)
    pred_confidence[:, AGENT_SLOT] = 0.5
    return WorldState(
        timestamp=torch.full((batch_size,), timestamp, device=device, dtype=dtype),
        latent_state=torch.zeros(batch_size, n, d, device=device, dtype=dtype),
        entity_xy=torch.zeros(batch_size, n, 2, device=device, dtype=dtype),
        entity_vel=torch.zeros(batch_size, n, 2, device=device, dtype=dtype),
        occupied=occupied,
        entity_id=entity_id,
        kind=kind,
        confidence=torch.zeros(batch_size, n, device=device, dtype=dtype),
        uncertainty=uncertainty,
        age_unobserved=torch.zeros(batch_size, n, device=device, dtype=dtype),
        xy_std=xy_std,
        vel_std=vel_std,
        existence=existence,
        pred_confidence=pred_confidence,
    )


class StateConstructor:
    """Bind current MinaUnits onto persistent world slots.

    Matching is by entity_id when present. Unobserved occupied slots persist
    until persistence.steps, then retire. This is architectural persistence,
    not a test-specific hardcode.

    Belief update: Observation + previous belief + memory + kinematics.
    Occupied is slot allocation. existence is P(this hypothesis is real).
    """

    def __init__(self, config: ArchitectureConfig) -> None:
        self.config = config

    def apply(
        self,
        units: MinaUnitBatch,
        previous: WorldState,
        positioned: Tensor,
        memory_hints: Tensor | None = None,
        experience_boost: tuple[Tensor, Tensor] | None = None,
    ) -> WorldState:
        """positioned: [B, N_obs, D]  observation latents after NPF + semantics."""
        units.validate()
        batch, n_obs, dim = positioned.shape
        n_slots = self.config.world_slots
        assert_shape("previous.latent_state", previous.latent_state, (batch, n_slots, dim))
        assert_finite("positioned", positioned)

        latent = previous.latent_state
        xy = previous.entity_xy
        vel = previous.entity_vel
        xy_std = previous.xy_std
        vel_std = previous.vel_std
        existence = previous.existence
        pred_confidence = previous.pred_confidence
        occupied = previous.occupied.clone()
        entity_id = previous.entity_id.clone()
        kind = previous.kind.clone()
        confidence = previous.confidence
        uncertainty = previous.uncertainty
        was_occupied = occupied.clone()
        age = previous.age_unobserved + was_occupied.to(dtype=previous.age_unobserved.dtype)

        updated = torch.zeros_like(occupied)
        dt = float(self.config.dt)
        corrections: list = []
        for b in range(batch):
            for i in range(n_obs):
                if not bool(units.mask[b, i]):
                    continue
                eid = int(units.entity_id[b, i].item())
                existed = bool(((entity_id[b] == eid) & occupied[b]).any()) if eid != 0 else False
                slot = self._find_or_allocate(entity_id[b], occupied[b], eid)
                observed_last = existed and float(previous.age_unobserved[b, slot].item()) == 0.0
                occupied[b, slot] = True
                entity_id[b, slot] = eid
                kind[b, slot] = units.kind[b, i]
                sel = torch.zeros_like(updated)
                sel[b, slot] = True
                sel_f = sel.unsqueeze(-1)
                ev_xy = units.spatial_position[b, i, :2]
                ev_vel = units.velocity[b, i]
                ev_age = (units.arrival_time[b, i] - units.timestamp[b, i]).clamp_min(0.0)
                if existed:
                    revision = revise_slot(
                        entity_id=eid,
                        old_xy=xy[b, slot],
                        old_vel=vel[b, slot],
                        old_confidence=confidence[b, slot],
                        old_uncertainty=uncertainty[b, slot],
                        evidence_xy=ev_xy,
                        evidence_vel=ev_vel,
                        evidence_confidence=units.confidence[b, i],
                        evidence_uncertainty=units.uncertainty[b, i],
                        belief_age_seconds=previous.age_unobserved[b, slot] * dt,
                        evidence_age_seconds=ev_age,
                        observed_last_cycle=observed_last,
                        evidence_source=f"source_{int(units.source_id[b, i].item())}",
                    )
                    xy = torch.where(sel_f, revision.xy.view(1, 1, 2).expand_as(xy), xy)
                    vel = torch.where(sel_f, revision.vel.view(1, 1, 2).expand_as(vel), vel)
                    confidence = torch.where(sel, revision.confidence.expand_as(confidence), confidence)
                    existence = torch.where(sel, revision.confidence.expand_as(existence), existence)
                    pred_confidence = torch.where(sel, revision.confidence.expand_as(pred_confidence), pred_confidence)
                    ev_std = units.uncertainty[b, i].clamp_min(BELIEF_STD_MIN).expand(2)
                    fused_xy_std = fuse(xy_std[b, slot], ev_std, revision.w_belief, revision.w_evidence)
                    fused_vel_std = fuse(vel_std[b, slot], ev_std, revision.w_belief, revision.w_evidence)
                    if revision.reason == "hypothesis_revision":
                        fused_xy_std = fused_xy_std + revision.conflict
                        fused_vel_std = fused_vel_std + revision.conflict
                    elif float(revision.w_evidence) < float(revision.w_belief):
                        fused_xy_std = fused_xy_std + 0.5 * revision.conflict
                        fused_vel_std = fused_vel_std + 0.5 * revision.conflict
                    xy_std = torch.where(
                        sel_f,
                        fused_xy_std.clamp_min(BELIEF_STD_MIN).view(1, 1, 2).expand_as(xy_std),
                        xy_std,
                    )
                    vel_std = torch.where(
                        sel_f,
                        fused_vel_std.clamp_min(BELIEF_STD_MIN).view(1, 1, 2).expand_as(vel_std),
                        vel_std,
                    )
                    if revision.event is not None:
                        corrections.append(revision.event)
                    conflict_fill = revision.conflict.view(1, 1).expand(uncertainty.shape[0], uncertainty.shape[1])
                    ch = uncertainty.clone()
                    ch[:, :, CONFLICT_CHANNEL] = torch.where(sel, conflict_fill, ch[:, :, CONFLICT_CHANNEL])
                    ch[:, :, NOISY_CHANNEL] = torch.where(
                        sel, units.uncertainty[b, i].expand_as(ch[:, :, NOISY_CHANNEL]), ch[:, :, NOISY_CHANNEL]
                    )
                    uncertainty = ch
                else:
                    xy = torch.where(sel_f, ev_xy.view(1, 1, 2).expand_as(xy), xy)
                    vel = torch.where(sel_f, ev_vel.view(1, 1, 2).expand_as(vel), vel)
                    confidence = torch.where(sel, units.confidence[b, i].expand_as(confidence), confidence)
                    existence = torch.where(
                        sel, units.confidence[b, i].clamp(0.5, 1.0).expand_as(existence), existence
                    )
                    pred_confidence = torch.where(sel, units.confidence[b, i].expand_as(pred_confidence), pred_confidence)
                    ev_std = units.uncertainty[b, i].clamp_min(BELIEF_STD_MIN).expand(2)
                    xy_std = torch.where(sel_f, ev_std.view(1, 1, 2).expand_as(xy_std), xy_std)
                    vel_std = torch.where(sel_f, ev_std.view(1, 1, 2).expand_as(vel_std), vel_std)
                    ch = uncertainty.clone()
                    ch[:, :, NOISY_CHANNEL] = torch.where(
                        sel, units.uncertainty[b, i].expand_as(ch[:, :, NOISY_CHANNEL]), ch[:, :, NOISY_CHANNEL]
                    )
                    uncertainty = ch
                latent = torch.where(sel_f, positioned[b, i].view(1, 1, dim).expand_as(latent), latent)
                age = torch.where(sel, torch.zeros_like(age), age)
                updated = updated | sel

        persist_horizon = age <= float(self.config.persistence.steps)
        if memory_hints is not None:
            assert_shape("memory_hints", memory_hints, (batch, n_slots, dim))
            inject = occupied & (~updated) & persist_horizon
            latent = torch.where(inject.unsqueeze(-1), 0.7 * latent + 0.3 * memory_hints, latent)
            prior = MEMORY_MEAN_GAIN * torch.tanh(memory_hints[..., :2])
            xy = torch.where(inject.unsqueeze(-1), xy + prior, xy)

        persist = occupied & (~updated) & persist_horizon
        retire = occupied & (~updated) & (age > float(self.config.persistence.steps))
        occupied = (occupied & (~retire)) | updated
        confidence = torch.where(persist, confidence * EXISTENCE_DECAY, confidence)
        existence = torch.where(
            persist,
            (existence * EXISTENCE_DECAY).clamp(min=BELIEF_EXISTENCE_FLOOR),
            existence,
        )
        pred_confidence = torch.where(persist, pred_confidence * PRED_CONF_DECAY, pred_confidence)
        extra = (age / float(max(self.config.persistence.steps, 1))).clamp(0.0, 1.0)
        uncertainty = torch.where(
            persist.unsqueeze(-1),
            (uncertainty + extra.unsqueeze(-1)).clamp(0.0, 1.0),
            uncertainty,
        )
        inflate = extra.unsqueeze(-1) * COAST_STD_GAIN
        xy_std = torch.where(persist.unsqueeze(-1), (xy_std + inflate).clamp_min(BELIEF_STD_MIN), xy_std)
        vel_std = torch.where(persist.unsqueeze(-1), (vel_std + inflate).clamp_min(BELIEF_STD_MIN), vel_std)
        if experience_boost is not None:
            xy_b, vel_b = experience_boost
            assert_shape("experience_xy_boost", xy_b, tuple(xy_std.shape))
            assert_shape("experience_vel_boost", vel_b, tuple(vel_std.shape))
            xy_std = torch.where(persist.unsqueeze(-1), (xy_std + xy_b).clamp_min(BELIEF_STD_MIN), xy_std)
            vel_std = torch.where(persist.unsqueeze(-1), (vel_std + vel_b).clamp_min(BELIEF_STD_MIN), vel_std)
        coast = persist.unsqueeze(-1).to(xy.dtype)
        xy = xy + vel * dt * coast
        entity_id = torch.where(occupied, entity_id, torch.zeros_like(entity_id))
        kind = torch.where(occupied, kind, torch.zeros_like(kind))
        stale_xy = retire.unsqueeze(-1).expand_as(xy)
        xy = torch.where(stale_xy, torch.zeros_like(xy), xy)
        vel = torch.where(stale_xy, torch.zeros_like(vel), vel)
        xy_std = torch.where(stale_xy, torch.ones_like(xy_std) * 0.1, xy_std)
        vel_std = torch.where(stale_xy, torch.ones_like(vel_std) * 0.1, vel_std)
        existence = torch.where(retire, torch.zeros_like(existence), existence)
        pred_confidence = torch.where(retire, torch.zeros_like(pred_confidence), pred_confidence)
        latent = torch.where(occupied.unsqueeze(-1), latent, torch.zeros_like(latent))
        now = units.timestamp.masked_fill(~units.mask, 0.0).max(dim=1).values
        return WorldState(
            timestamp=now,
            latent_state=latent,
            entity_xy=xy,
            entity_vel=vel,
            occupied=occupied,
            entity_id=entity_id,
            kind=kind,
            confidence=confidence,
            uncertainty=uncertainty,
            age_unobserved=torch.where(occupied, age, torch.zeros_like(age)),
            xy_std=xy_std,
            vel_std=vel_std,
            existence=existence,
            pred_confidence=pred_confidence,
            corrections=tuple(corrections),
        )

    def _find_or_allocate(self, ids: Tensor, occupied: Tensor, eid: int) -> int:
        matches = (ids == eid) & occupied
        if bool(matches.any()) and eid != 0:
            return int(matches.nonzero(as_tuple=False)[0].item())
        free = (~occupied).nonzero(as_tuple=False)
        if free.numel() == 0:
            ages = occupied.to(torch.float32)
            ages[AGENT_SLOT] = -1.0
            return int(torch.argmax(ages).item())
        slot = int(free[0].item())
        if slot == AGENT_SLOT and eid != 1:
            if free.numel() > 1:
                return int(free[1].item())
        return slot
