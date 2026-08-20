"""Split / merge tensor maps for sharded *.mina payloads."""

from __future__ import annotations

from typing import Any

from torch import Tensor


def tensor_nbytes(value: Any) -> int:
    if isinstance(value, Tensor):
        return int(value.numel() * value.element_size())
    return 0


def split_tensor_map(state: dict[str, Any], max_bytes: int) -> list[dict[str, Any]]:
    if max_bytes < 1:
        raise ValueError("max_bytes must be >= 1")
    shards: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    used = 0
    for key, value in state.items():
        size = tensor_nbytes(value)
        if current and used + size > max_bytes:
            shards.append(current)
            current = {}
            used = 0
        current[key] = value
        used += size
    if current:
        shards.append(current)
    return shards


def merge_tensor_maps(shards: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for shard in shards:
        overlap = set(out) & set(shard)
        if overlap:
            raise ValueError(f"duplicate shard keys: {sorted(overlap)[:8]}")
        out.update(shard)
    return out
