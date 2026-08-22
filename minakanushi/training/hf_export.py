"""Public Hugging Face safetensors mirror. Canonical runtime stays *.mina.

Does not construct MinakanushiSystem. Reads tensor maps out of the zip and
writes bf16 shards plus native identity JSON. Optimizer / RNG never leave
the .mina.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Iterator

import torch
import yaml
from torch import Tensor

from minakanushi.training.checkpoint import CONFIG_NAME, MANIFEST_NAME, WEIGHTS_NAME
from minakanushi.training.shard import tensor_nbytes

FORBIDDEN_MODEL_TYPES = {"llama", "gemma", "qwen", "mistral", "gpt2", "gpt-neox", "gpt_neox"}
FORBIDDEN_CLASS_NAMES = (
    "LlamaForCausalLM",
    "GemmaForCausalLM",
    "Qwen2ForCausalLM",
    "MistralForCausalLM",
    "GPT2LMHeadModel",
)


def assert_not_llm_card(payload: dict) -> None:
    if payload.get("architectures"):
        raise ValueError("architectures is a Transformers field; MINA does not use it")
    model_type = str(payload.get("model_type") or "").lower()
    if model_type in FORBIDDEN_MODEL_TYPES:
        raise ValueError(f"model_type={model_type!r} is a lie")
    if payload.get("chat_template"):
        raise ValueError("chat_template is forbidden on a MINA card")
    blob = json.dumps(payload, ensure_ascii=True)
    for marker in FORBIDDEN_CLASS_NAMES:
        if marker in blob:
            raise ValueError(f"HF mirror card must not contain {marker!r}")

DEFAULT_SHARD_BYTES = 5 * 1024 * 1024 * 1024
MIRROR_DTYPE = "bf16"


def _require_safetensors():
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise SystemExit(
            "safetensors is required for the HF mirror extra: pip install '.[hf]'"
        ) from exc
    return save_file


def hf_config(manifest: dict, *, canonical_checkpoint: str) -> dict:
    config = {
        "model_type": "minakanushi",
        "architecture": "MINAKANUSHI",
        "organization": manifest.get("organization", "NULLXES"),
        "short_name": "MINA",
        "parameters": int(manifest.get("parameters") or _formula_params(manifest)),
        "format": "safetensors_mirror",
        "runtime": "load_mina",
        "world_model": True,
        "action_output": "ActionIntent",
        "native_runtime": "nullxes",
        "system_class": manifest.get("system_class", "adaptive_situational_intelligence"),
        "architecture_generation": manifest.get("generation", 1),
        "architecture_version": str(manifest.get("architecture_version", "0.1")),
        "latent_dim": int(manifest["latent_dim"]),
        "state_dim": int(manifest["state_dim"]),
        "memory_dim": int(manifest["memory_dim"]),
        "core_depth": int(manifest["core_depth"]),
        "world_slots": int(manifest["world_slots"]),
        "memory_slots": int(manifest["memory_slots"]),
        "uncertainty_channels": int(manifest["uncertainty_channels"]),
        "future_branches": int(manifest["future_branches"]),
        "canonical_checkpoint": canonical_checkpoint,
        "canonical_format": "mina",
        "pwm": False,
        "not_a_language_model": True,
        "not_a_chat_model": True,
        "auto_map": {
            "AutoConfig": "minakanushi.hub.MinakanushiHFConfig",
            "AutoModel": "minakanushi.hub.MinakanushiHubModel",
        },
    }
    assert_not_llm_card(config)
    return config


def minakanushi_native_config(manifest: dict, *, canonical_checkpoint: str) -> dict:
    """Native identity next to the Hub card. Not a Transformers architecture list."""
    config = {
        "architecture": "MINAKANUSHI",
        "organization": manifest.get("organization", "NULLXES"),
        "short_name": "MINA",
        "system_class": manifest.get("system_class", "adaptive_situational_intelligence"),
        "native_runtime": "nullxes",
        "canonical_format": "mina",
        "canonical_checkpoint": canonical_checkpoint,
        "public_mirror": "safetensors",
        "action_output": "ActionIntent",
        "pwm": False,
        "latent_dim": int(manifest["latent_dim"]),
        "state_dim": int(manifest["state_dim"]),
        "memory_dim": int(manifest["memory_dim"]),
        "core_depth": int(manifest["core_depth"]),
        "world_slots": int(manifest["world_slots"]),
        "memory_slots": int(manifest["memory_slots"]),
    }
    assert_not_llm_card(config)
    return config


def _formula_params(manifest: dict) -> int:
    extra = manifest.get("train") or {}
    report = extra.get("parameter_report") if isinstance(extra, dict) else None
    if isinstance(report, dict) and "total" in report:
        return int(report["total"])
    return 6_799_130_646


def minakanushi_card(*, hardware: str = "B300/H200 class") -> dict:
    card = {
        "name": "NULLXES MINAKANUSHI 6.8B",
        "type": "Autonomous World Intelligence",
        "architecture": "MINAKANUSHI",
        "organization": "NULLXES",
        "short_name": "MINA",
        "not_a_chat_model": True,
        "not_a_language_model": True,
        "outputs": "ActionIntent",
        "hardware": hardware,
        "canonical_format": "mina",
        "public_mirror": "safetensors",
    }
    assert_not_llm_card(card)
    return card


def runtime_card(canonical_checkpoint: str) -> dict:
    card = {
        "runtime": "load_mina",
        "construct": "MinakanushiSystem",
        "canonical_format": "mina",
        "canonical_checkpoint": canonical_checkpoint,
        "public_mirror": "safetensors",
        "action_output": "ActionIntent",
        "pwm": False,
        "transformers_auto_model": False,
        "transformers_auto_config": True,
        "transformers_hub_role": "type_tag_only",
        "generation": False,
        "chat_template": False,
    }
    assert_not_llm_card(card)
    return card


def _to_mirror_tensor(value: Tensor) -> Tensor:
    tensor = value.detach().to("cpu").contiguous().clone()
    if tensor.is_floating_point():
        return tensor.to(dtype=torch.bfloat16)
    return tensor


def _iter_system_shards(zf: zipfile.ZipFile) -> Iterator[dict[str, Any]]:
    names = set(zf.namelist())
    if WEIGHTS_NAME in names:
        payload = torch.load(io.BytesIO(zf.read(WEIGHTS_NAME)), map_location="cpu", weights_only=False)
        system = payload.get("system") if isinstance(payload, dict) else None
        if not isinstance(system, dict):
            raise ValueError("weights.pt has no system tensor map")
        yield system
        return
    shard_names = sorted(n for n in names if n.startswith("weights/system-") and n.endswith(".pt"))
    if not shard_names:
        raise ValueError("checkpoint has neither weights.pt nor sharded system maps")
    for name in shard_names:
        shard = torch.load(io.BytesIO(zf.read(name)), map_location="cpu", weights_only=False)
        if not isinstance(shard, dict):
            raise ValueError(f"{name} is not a tensor map")
        yield shard


def _flush_shard(tensors: dict[str, Tensor], path: Path, save_file) -> int:
    if not tensors:
        raise ValueError("refusing empty safetensors shard")
    save_file(
        tensors,
        str(path),
        metadata={
            "format": "pt",
            "architecture": "MINAKANUSHI",
            "mirror": "safetensors_mirror",
            "canonical_format": "mina",
        },
    )
    return sum(int(t.numel()) for t in tensors.values() if t.is_floating_point())


def write_cards(out_dir: Path, manifest: dict, canonical_checkpoint: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "config.json": hf_config(manifest, canonical_checkpoint=canonical_checkpoint),
        "minakanushi_config.json": minakanushi_native_config(
            manifest, canonical_checkpoint=canonical_checkpoint
        ),
        "MINAKANUSHI_CARD.json": minakanushi_card(),
        "minakanushi_runtime.json": runtime_card(canonical_checkpoint),
    }
    for name, payload in files.items():
        (out_dir / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    gen = out_dir / "generation"
    gen.mkdir(exist_ok=True)
    (gen / "NO").write_text(
        "MINAKANUSHI does not generate tokens.\n"
        "No chat template.\n"
        "No generation_config.json.\n"
        "Output is ActionIntent.\n",
        encoding="utf-8",
    )


def export_hf_mirror(
    mina_path: str | Path,
    out_dir: str | Path,
    *,
    shard_bytes: int = DEFAULT_SHARD_BYTES,
    cards_only: bool = False,
    readme: str | Path | None = None,
    license_path: str | Path | None = None,
) -> dict:
    """Export a safetensors mirror next to identity JSON. Does not upload."""
    mina_path = Path(mina_path)
    out_dir = Path(out_dir)
    if mina_path.suffix != ".mina":
        raise ValueError(f"canonical checkpoint must be *.mina, got {mina_path}")
    if shard_bytes < 1:
        raise ValueError("shard_bytes must be >= 1")
    save_file = None if cards_only else _require_safetensors()
    with zipfile.ZipFile(mina_path, "r") as zf:
        if MANIFEST_NAME not in zf.namelist():
            raise ValueError("checkpoint missing manifest.yaml")
        manifest = yaml.safe_load(zf.read(MANIFEST_NAME))
        if manifest.get("architecture") != "MINAKANUSHI":
            raise ValueError("refusing non-MINAKANUSHI checkpoint")
        if CONFIG_NAME in zf.namelist():
            (out_dir / "architecture.yaml").parent.mkdir(parents=True, exist_ok=True)
        write_cards(out_dir, manifest, canonical_checkpoint=mina_path.name)
        if CONFIG_NAME in zf.namelist():
            (out_dir / "architecture.yaml").write_bytes(zf.read(CONFIG_NAME))
        if cards_only:
            _copy_readme(readme, out_dir)
            _copy_license(license_path, out_dir)
            return {"cards_only": True, "canonical": mina_path.name, "out": str(out_dir)}

        current: dict[str, Tensor] = {}
        used = 0
        tmp_shards: list[tuple[Path, list[str]]] = []
        float_params = 0
        for system_map in _iter_system_shards(zf):
            for key, value in system_map.items():
                if not isinstance(value, Tensor):
                    raise ValueError(f"system[{key!r}] is {type(value).__name__}, not a Tensor")
                mirrored = _to_mirror_tensor(value)
                size = tensor_nbytes(mirrored)
                if current and used + size > shard_bytes:
                    tmp = out_dir / f"_shard_{len(tmp_shards):05d}.safetensors"
                    float_params += _flush_shard(current, tmp, save_file)
                    tmp_shards.append((tmp, list(current)))
                    current = {}
                    used = 0
                current[key] = mirrored
                used += size
        if current:
            tmp = out_dir / f"_shard_{len(tmp_shards):05d}.safetensors"
            float_params += _flush_shard(current, tmp, save_file)
            tmp_shards.append((tmp, list(current)))

    if not tmp_shards:
        raise ValueError("no system tensors to export")
    n = len(tmp_shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    final_names: list[str] = []
    for i, (tmp, keys) in enumerate(tmp_shards, start=1):
        name = f"model-{i:05d}-of-{n:05d}.safetensors"
        dest = out_dir / name
        tmp.replace(dest)
        final_names.append(name)
        total_size += dest.stat().st_size
        for key in keys:
            weight_map[key] = name

    index = {
        "metadata": {
            "total_size": total_size,
            "format": "safetensors_mirror",
            "architecture": "MINAKANUSHI",
            "canonical_format": "mina",
            "canonical_checkpoint": mina_path.name,
            "dtype": MIRROR_DTYPE,
            "parameters": float_params,
        },
        "weight_map": weight_map,
    }
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )
    _copy_readme(readme, out_dir)
    _copy_license(license_path, out_dir)
    return {
        "cards_only": False,
        "canonical": mina_path.name,
        "out": str(out_dir),
        "shards": final_names,
        "parameters": float_params,
        "bytes": total_size,
    }


def _copy_readme(readme: str | Path | None, out_dir: Path) -> None:
    if readme is None:
        return
    src = Path(readme)
    if src.exists():
        (out_dir / "README.md").write_bytes(src.read_bytes())


def _copy_license(license_path: str | Path | None, out_dir: Path) -> None:
    if license_path is None:
        return
    src = Path(license_path)
    if src.exists():
        (out_dir / "LICENSE").write_bytes(src.read_bytes())
