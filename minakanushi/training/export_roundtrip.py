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


def _load_shards(out_dir: Path) -> dict[str, Tensor]:
    from safetensors.torch import load_file

    restored: dict[str, Tensor] = {}
    index_path = out_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for name in sorted(set(index["weight_map"].values())):
            restored.update(load_file(str(out_dir / name)))
        return restored
    shards = sorted(out_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors under {out_dir}")
    for path in shards:
        restored.update(load_file(str(path)))
    return restored


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
