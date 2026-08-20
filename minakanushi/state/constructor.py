"""WorldState factory and empty-state constructor."""

from __future__ import annotations

import torch
from torch import Tensor

from minakanushi.architecture.config import ArchitectureConfig
from minakanushi.architecture.mina_unit import KIND_IDS, MinaUnitBatch
from minakanushi.architecture.outputs import PositionState
from minakanushi.state.entity import AGENT_SLOT
from minakanushi.state.world import WorldState
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
    )


class StateConstructor:
    """Bind current MinaUnits onto persistent world slots.

    Matching is by entity_id when present. Unobserved occupied slots persist
    until persistence.steps, then retire. This is architectural persistence,
    not a test-specific hardcode.
    """

    def __init__(self, config: ArchitectureConfig) -> None:
        self.config = config

    def apply(
        self,
        units: MinaUnitBatch,
        previous: WorldState,
        positioned: Tensor,
        memory_hints: Tensor | None = None,
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
        occupied = previous.occupied.clone()
        entity_id = previous.entity_id.clone()
        kind = previous.kind.clone()
        confidence = previous.confidence
        uncertainty = previous.uncertainty
        was_occupied = occupied.clone()
        age = previous.age_unobserved + was_occupied.to(dtype=previous.age_unobserved.dtype)

        updated = torch.zeros_like(occupied)
        for b in range(batch):
            for i in range(n_obs):
                if not bool(units.mask[b, i]):
                    continue
                eid = int(units.entity_id[b, i].item())
                slot = self._find_or_allocate(entity_id[b], occupied[b], eid)
                occupied[b, slot] = True
                entity_id[b, slot] = eid
                kind[b, slot] = units.kind[b, i]
                sel = torch.zeros_like(updated)
                sel[b, slot] = True
                sel_f = sel.unsqueeze(-1)
                xy = torch.where(sel_f, units.spatial_position[b, i, :2].view(1, 1, 2).expand_as(xy), xy)
                latent = torch.where(sel_f, positioned[b, i].view(1, 1, dim).expand_as(latent), latent)
                confidence = torch.where(sel, units.confidence[b, i].expand_as(confidence), confidence)
                noise = units.uncertainty[b, i].clamp_min(0.0).view(1, 1, 1).expand_as(uncertainty)
                uncertainty = torch.where(sel_f, noise, uncertainty)
                age = torch.where(sel, torch.zeros_like(age), age)
                updated = updated | sel

        if memory_hints is not None:
            assert_shape("memory_hints", memory_hints, (batch, n_slots, dim))
            inject = occupied & (~updated) & (age <= self.config.persistence.steps)
            latent = torch.where(inject.unsqueeze(-1), 0.7 * latent + 0.3 * memory_hints, latent)

        persist = occupied & (~updated) & (age <= float(self.config.persistence.steps))
        retire = occupied & (~updated) & (age > float(self.config.persistence.steps))
        occupied = (occupied & (~retire)) | updated
        confidence = torch.where(persist, confidence * 0.85, confidence)
        extra = (age / float(max(self.config.persistence.steps, 1))).clamp(0.0, 1.0)
        uncertainty = torch.where(
            persist.unsqueeze(-1),
            (uncertainty + extra.unsqueeze(-1)).clamp(0.0, 1.0),
            uncertainty,
        )
        entity_id = torch.where(occupied, entity_id, torch.zeros_like(entity_id))
        kind = torch.where(occupied, kind, torch.zeros_like(kind))
        stale_xy = retire.unsqueeze(-1).expand_as(xy)
        xy = torch.where(stale_xy, torch.zeros_like(xy), xy)
        vel = torch.where(stale_xy, torch.zeros_like(vel), vel)
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
        )

    def _find_or_allocate(self, ids: Tensor, occupied: Tensor, eid: int) -> int:
        matches = (ids == eid) & occupied
        if bool(matches.any()) and eid != 0:
            return int(matches.nonzero(as_tuple=False)[0].item())
        free = (~occupied).nonzero(as_tuple=False)
        if free.numel() == 0:
            # retire the oldest unobserved non-agent slot
            ages = occupied.to(torch.float32)
            ages[AGENT_SLOT] = -1.0
            return int(torch.argmax(ages).item())
        slot = int(free[0].item())
        if slot == AGENT_SLOT and eid != 1:
            if free.numel() > 1:
                return int(free[1].item())
        return slot
