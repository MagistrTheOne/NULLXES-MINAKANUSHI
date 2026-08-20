"""Native *.mina checkpoint: identity + config + weights + optimizer."""

from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from torch.optim import Optimizer

from minakanushi.architecture.config import ArchitectureConfig
from minakanushi.architecture.model import MinakanushiSystem
from minakanushi.training.parallel import apply_full_checkpoint, dist_barrier, is_rank0
from minakanushi.training.shard import merge_tensor_maps, split_tensor_map


MANIFEST_NAME = "manifest.yaml"
WEIGHTS_NAME = "weights.pt"
CONFIG_NAME = "architecture.yaml"
CHECKPOINT_FORMAT_VERSION = 2
_STEP_IN_NAME = re.compile(r"step(\d+)", re.IGNORECASE)


def _require(manifest: dict, key: str, expected) -> None:
    if key not in manifest:
        raise ValueError(f"checkpoint missing '{key}'")
    if manifest[key] != expected:
        raise ValueError(f"checkpoint {key}={manifest[key]!r} incompatible with {expected!r}")


def build_manifest(config: ArchitectureConfig, extras: dict | None = None) -> dict:
    identity = config.identity
    manifest = {
        "format": "nullxes-minakanushi",
        "architecture": identity.architecture,
        "organization": identity.organization,
        "generation": identity.architecture_generation,
        "architecture_version": identity.architecture_version,
        "checkpoint_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "system_class": identity.system_class,
        "native_runtime": identity.native_runtime,
        "latent_dim": config.latent_dim,
        "state_dim": config.state_dim,
        "memory_dim": config.memory_dim,
        "world_slots": config.world_slots,
        "memory_slots": config.memory_slots,
        "core_depth": config.core_depth,
        "uncertainty_channels": config.uncertainty_channels,
        "future_branches": config.future_branches,
        "modules": {
            "position_field": True,
            "world_core": True,
            "memory": True,
            "uncertainty": True,
            "future_engine": True,
            "strategy_engine": True,
            "constraint_kernel": True,
            "self_model": True,
            "authority": True,
            "runtime": True,
        },
    }
    if extras:
        manifest["train"] = extras
    return manifest


def _identity_json(extras: dict | None) -> str:
    return json.dumps(
        {
            "architecture": "MINAKANUSHI",
            "short_name": "MINA",
            "architecture_id": "nullxes.minakanushi",
            "organization": "NULLXES",
            "native_runtime": "nullxes",
            "identity_state": (extras or {}).get("identity"),
        }
    )


def _write_zip_bytes(zf: zipfile.ZipFile, name: str, payload: object) -> None:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    zf.writestr(name, buffer.getvalue())


def save_mina(
    path: str | Path,
    system: MinakanushiSystem,
    *,
    optimizer: Optimizer | None = None,
    extras: dict | None = None,
    tensors: dict | None = None,
    shard_max_bytes: int = 0,
    gathered: dict | None = None,
) -> Path:
    """Write *.mina from a full CPU payload.

    When torch.distributed is initialized, only rank 0 writes. Callers that wrap
    with FSDP2 must pass `gathered` from collect_full_checkpoint() so this does
    not persist a local shard via system.state_dict().
    """
    path = Path(path)
    if path.suffix != ".mina":
        raise ValueError(f"checkpoint must use .mina suffix, got {path}")
    if not is_rank0():
        dist_barrier()
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if gathered is None:
        payload = {
            "system": system.state_dict(),
            "parameter_report": system.parameter_report(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "runtime": tensors,
        }
    else:
        payload = {
            "system": gathered["system"],
            "parameter_report": gathered.get("parameter_report") or system.parameter_report(),
            "optimizer": gathered.get("optimizer"),
            "runtime": tensors if tensors is not None else gathered.get("runtime"),
        }
    manifest = build_manifest(system.config, extras)
    sharded = int(shard_max_bytes) > 0
    manifest["sharded"] = sharded
    manifest["fsdp_gathered"] = bool(gathered is not None and gathered.get("gathered"))
    if sharded:
        system_shards = split_tensor_map(payload["system"], int(shard_max_bytes))
        manifest["n_system_shards"] = len(system_shards)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_NAME, yaml.safe_dump(manifest, sort_keys=False))
        zf.writestr(CONFIG_NAME, yaml.safe_dump(asdict(system.config), sort_keys=False))
        zf.writestr("identity.json", _identity_json(extras))
        if sharded:
            for i, shard in enumerate(system_shards):
                _write_zip_bytes(zf, f"weights/system-{i:05d}.pt", shard)
            rest = {k: v for k, v in payload.items() if k != "system"}
            _write_zip_bytes(zf, "weights/sidecar.pt", rest)
        else:
            _write_zip_bytes(zf, WEIGHTS_NAME, payload)
    dist_barrier()
    return path


def load_mina(
    path: str | Path,
    system: MinakanushiSystem,
    *,
    optimizer: Optimizer | None = None,
    return_payload: bool = False,
) -> dict | tuple[dict, dict]:
    path = Path(path)
    cfg = system.config
    with zipfile.ZipFile(path, "r") as zf:
        manifest = yaml.safe_load(zf.read(MANIFEST_NAME))
        _require(manifest, "architecture", "MINAKANUSHI")
        _require(manifest, "organization", "NULLXES")
        _require(manifest, "native_runtime", "nullxes")
        _require(manifest, "architecture_version", cfg.identity.architecture_version)
        for key, expected in {
            "latent_dim": cfg.latent_dim,
            "state_dim": cfg.state_dim,
            "memory_dim": cfg.memory_dim,
            "world_slots": cfg.world_slots,
            "memory_slots": cfg.memory_slots,
            "core_depth": cfg.core_depth,
            "uncertainty_channels": cfg.uncertainty_channels,
            "future_branches": cfg.future_branches,
        }.items():
            if int(manifest[key]) != int(expected):
                raise ValueError(f"checkpoint {key}={manifest[key]} vs system {expected}")
        names = set(zf.namelist())
        if WEIGHTS_NAME in names:
            payload = torch.load(
                io.BytesIO(zf.read(WEIGHTS_NAME)),
                map_location="cpu",
                weights_only=False,
            )
        else:
            shard_names = sorted(n for n in names if n.startswith("weights/system-") and n.endswith(".pt"))
            if not shard_names:
                raise ValueError("checkpoint has neither weights.pt nor sharded system maps")
            shards = [
                torch.load(io.BytesIO(zf.read(name)), map_location="cpu", weights_only=False)
                for name in shard_names
            ]
            sidecar = torch.load(
                io.BytesIO(zf.read("weights/sidecar.pt")),
                map_location="cpu",
                weights_only=False,
            )
            payload = dict(sidecar)
            payload["system"] = merge_tensor_maps(shards)
    apply_full_checkpoint(system, optimizer, payload)
    if return_payload:
        return manifest, payload
    return manifest


def checkpoint_step(path: str | Path) -> int:
    """Training step encoded in `*_step{N}.mina`. Missing step sorts as -1."""
    match = _STEP_IN_NAME.search(Path(path).stem)
    if match is None:
        return -1
    return int(match.group(1))


def latest_mina(directory: str | Path) -> Path:
    """Newest checkpoint by numeric step, not lexicographic filename order."""
    directory = Path(directory)
    files = list(directory.glob("*.mina"))
    if not files:
        raise FileNotFoundError(f"no .mina checkpoint in {directory}")
    return max(files, key=checkpoint_step)
