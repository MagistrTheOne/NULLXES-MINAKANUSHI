"""Serialize WorldState tensors for runtime checkpoints. Not a training dump."""

from __future__ import annotations

import torch

from minakanushi.state.world import WorldState

WORLD_TENSOR_FIELDS: tuple[str, ...] = (
    "timestamp",
    "latent_state",
    "entity_xy",
    "entity_vel",
    "occupied",
    "entity_id",
    "kind",
    "confidence",
    "uncertainty",
    "age_unobserved",
    "xy_std",
    "vel_std",
    "existence",
    "pred_confidence",
)


def dump_world(world: WorldState) -> dict:
    return {
        "tensors": {name: getattr(world, name).detach().cpu().clone() for name in WORLD_TENSOR_FIELDS},
        "self_index": int(world.self_index),
        "provenance": str(world.provenance),
    }


def load_world(blob: dict | None, *, device: torch.device) -> WorldState | None:
    if not blob:
        return None
    tensors = {name: blob["tensors"][name].to(device=device) for name in WORLD_TENSOR_FIELDS}
    return WorldState(
        **tensors,
        self_index=int(blob.get("self_index", 0)),
        provenance=str(blob.get("provenance", "checkpoint")),
        corrections=(),
    )


def belief_fingerprint(world: WorldState) -> str:
    occ = world.occupied[0]
    if not bool(occ.any()):
        return "empty"
    xy = float(world.entity_xy[0, occ].sum().item())
    exist = float(world.existence[0, occ].sum().item())
    return f"n={int(occ.sum().item())}:xy={xy:.6f}:ex={exist:.6f}"
