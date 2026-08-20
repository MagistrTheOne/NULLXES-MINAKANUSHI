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


def save_mina(
    path: str | Path,
    system: MinakanushiSystem,
    *,
    optimizer: Optimizer | None = None,
    extras: dict | None = None,
    tensors: dict | None = None,
) -> Path:
    path = Path(path)
    if path.suffix != ".mina":
        raise ValueError(f"checkpoint must use .mina suffix, got {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(system.config, extras)
    payload = {
        "system": system.state_dict(),
        "parameter_report": system.parameter_report(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "runtime": tensors,
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_NAME, yaml.safe_dump(manifest, sort_keys=False))
        zf.writestr(CONFIG_NAME, yaml.safe_dump(asdict(system.config), sort_keys=False))
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        zf.writestr(WEIGHTS_NAME, buffer.getvalue())
        zf.writestr(
            "identity.json",
            json.dumps(
                {
                    "architecture": "MINAKANUSHI",
                    "short_name": "MINA",
                    "architecture_id": "nullxes.minakanushi",
                    "organization": "NULLXES",
                    "native_runtime": "nullxes",
                    "identity_state": (extras or {}).get("identity"),
                }
            ),
        )
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
        payload = torch.load(io.BytesIO(zf.read(WEIGHTS_NAME)), map_location="cpu")
    incompatible = system.load_state_dict(payload["system"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"strict load failed: {incompatible}")
    if optimizer is not None:
        if payload.get("optimizer") is None:
            raise ValueError("checkpoint has no optimizer state; refusing silent resume")
        optimizer.load_state_dict(payload["optimizer"])
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
