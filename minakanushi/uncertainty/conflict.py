"""Conflict between evidence sources. Distinct from missing or noisy evidence."""

from __future__ import annotations

import torch
from torch import Tensor

from minakanushi.architecture.mina_unit import MinaUnitBatch
from minakanushi.state.world import WorldState


def conflict_score(world: WorldState, units: MinaUnitBatch) -> Tensor:
    """Per-slot conflict in [0, 1]. [B, N_world]."""
    scores = torch.zeros(world.occupied.shape, device=world.latent_state.device, dtype=world.latent_state.dtype)
    for b in range(world.entity_id.shape[0]):
        for i in range(units.mask.shape[1]):
            if not bool(units.mask[b, i]):
                continue
            eid = units.entity_id[b, i]
            slots = (world.entity_id[b] == eid) & world.occupied[b]
            if not bool(slots.any()):
                continue
            slot = int(slots.nonzero(as_tuple=False)[0].item())
            predicted = world.entity_xy[b, slot]
            observed = units.spatial_position[b, i, :2]
            dist = torch.linalg.vector_norm(predicted - observed)
            value = torch.clamp(dist / 2.0, 0.0, 1.0)
            sel = torch.zeros_like(scores, dtype=torch.bool)
            sel[b, slot] = True
            scores = torch.where(sel, value.expand_as(scores), scores)
    return scores
