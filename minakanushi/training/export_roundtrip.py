"""Export Hugging Face safetensors from a *.mina and load the shards back.

Does not construct 6.8B. Canonical runtime stays load_mina.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from minakanushi.training.hf_export import export_hf_mirror


def _shard_names(out_dir: Path) -> list[str]:
    index_path = out_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        return sorted(set(index["weight_map"].values()))
    return [path.name for path in sorted(out_dir.glob("*.safetensors"))]


def _load_shards(out_dir: Path) -> dict[str, Tensor]:
    from safetensors.torch import load_file

    names = _shard_names(out_dir)
    if not names:
        raise FileNotFoundError(f"no safetensors under {out_dir}")
    restored: dict[str, Tensor] = {}
    for name in names:
        restored.update(load_file(str(out_dir / name)))
    return restored


def shard_header_shapes(out_dir: Path) -> dict[str, tuple[int, ...]]:
    """Shapes from safetensors headers. Empty tuple is a valid 0-dim scalar."""
    from safetensors import safe_open

    shapes: dict[str, tuple[int, ...]] = {}
    for name in _shard_names(out_dir):
        with safe_open(str(out_dir / name), framework="pt") as fh:
            for key in fh.keys():
                shapes[key] = tuple(fh.get_slice(key).get_shape())
    return shapes


def tensors_match_headers(
    tensors: dict[str, Tensor],
    shapes: dict[str, tuple[int, ...]],
) -> bool:
    """True when every loaded tensor matches its shard header, including ndim==0."""
    if set(tensors) != set(shapes):
        return False
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            return False
        if tuple(tensor.shape) != shapes[name]:
            return False
    return True


def export_and_reload(
    mina: str | Path,
    out_dir: str | Path,
    *,
    shard_bytes: int = 64,
    cards_only: bool = False,
) -> dict[str, Any]:
    mina = Path(mina)
    out_dir = Path(out_dir)
    result = export_hf_mirror(mina, out_dir, shard_bytes=shard_bytes, cards_only=cards_only)
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "minakanushi":
        raise ValueError("mirror card is not minakanushi")
    if "LlamaForCausalLM" in json.dumps(config):
        raise ValueError("mirror must not look like CausalLM")
    if cards_only:
        return {**result, "reloaded": False, "n_tensors": 0}
    restored = _load_shards(out_dir)
    if not restored:
        raise ValueError("safetensors reload produced no tensors")
    for tensor in restored.values():
        if tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
            raise ValueError(f"mirror dtype {tensor.dtype} is not bf16")
    return {**result, "reloaded": True, "n_tensors": len(restored)}
