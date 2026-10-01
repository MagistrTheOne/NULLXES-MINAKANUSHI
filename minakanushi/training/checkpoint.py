"""Native *.mina checkpoint v3: SAFETENSORS-INSIDE / PICKLE-OUT.

Layout (zip, STORED): manifest.yaml, architecture.yaml, identity.json,
weights/tensors_index.json, weights/system-*.safetensors,
weights/optimizer.safetensors (optional), weights/runtime.safetensors
(optional), weights/sidecar.json.

No torch.save / torch.load on the .mina path. Legacy *.pt payloads reject.
Fail-before-mutation: hash -> schema -> tensors -> shapes, then apply.
"""

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
from minakanushi.training.mina_format import (
    CHECKPOINT_FORMAT_VERSION,
    CONFIG_NAME,
    IDENTITY_NAME,
    MANIFEST_NAME,
    OPTIMIZER_NAME,
    RUNTIME_TENSORS_NAME,
    SIDECAR_NAME,
    SYSTEM_SHARD_PREFIX,
    SYSTEM_SHARD_SUFFIX,
    TENSORS_INDEX_NAME,
    _SHARD_RE,
    check_namelist,
    decode_runtime,
    encode_runtime,
    flatten_optimizer,
    load_safetensors_bytes,
    safetensors_bytes,
    sha256_bytes,
    unflatten_optimizer,
)
from minakanushi.training.parallel import apply_full_checkpoint, dist_barrier, is_rank0
from minakanushi.training.shard import merge_tensor_maps, split_tensor_map

_STEP_IN_NAME = re.compile(r"step(\d+)", re.IGNORECASE)


def yaml_safe(value):
    """Manifest extras must be SafeDumper-portable. Tuples are not a YAML type."""
    if isinstance(value, tuple):
        return [yaml_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): yaml_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [yaml_safe(item) for item in value]
    return value


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
        manifest["train"] = yaml_safe(extras)
    return manifest


def _identity_json(extras: dict | None) -> str:
    return json.dumps(
        {
            "architecture": "MINAKANUSHI",
            "short_name": "MINA",
            "architecture_id": "nullxes.minakanushi",
            "organization": "NULLXES",
            "native_runtime": "nullxes",
            "identity_state": yaml_safe((extras or {}).get("identity")),
        }
    )


def _parameter_report(system: MinakanushiSystem, gathered: dict | None) -> dict:
    if gathered is not None and gathered.get("parameter_report"):
        report = gathered["parameter_report"]
        return {"total": int(report["total"]), "trainable": int(report.get("trainable", report["total"]))}
    try:
        report = system.parameter_report()
        return {"total": int(report["total"]), "trainable": int(report.get("trainable", report["total"]))}
    except (AttributeError, KeyError, TypeError, ValueError):
        return {"total": 0, "trainable": 0}


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
    """Write *.mina v3 from a full CPU payload (safetensors inside).

    When torch.distributed is initialized, only rank 0 writes. Callers that wrap
    with FSDP2 must pass `gathered` from collect_full_checkpoint().
    """
    path = Path(path)
    if path.suffix != ".mina":
        raise ValueError(f"checkpoint must use .mina suffix, got {path}")
    if not is_rank0():
        dist_barrier()
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if gathered is None:
        system_map = {k: v.detach().cpu().contiguous() for k, v in system.state_dict().items()}
        opt_state = optimizer.state_dict() if optimizer is not None else None
    else:
        system_map = {k: v.detach().cpu().contiguous() for k, v in gathered["system"].items()}
        opt_state = gathered.get("optimizer")
    bool_fields = sorted(k for k, v in system_map.items() if v.dtype == torch.bool)
    for k in bool_fields:
        system_map[k] = system_map[k].to(dtype=torch.uint8)
    opt_tensors, opt_meta = flatten_optimizer(opt_state)
    runtime_tensors, runtime_meta = encode_runtime(tensors)
    max_bytes = int(shard_max_bytes)
    if max_bytes > 0:
        system_shards = split_tensor_map(system_map, max_bytes)
    else:
        system_shards = [system_map]
    manifest = build_manifest(system.config, extras)
    manifest["sharded"] = max_bytes > 0
    manifest["fsdp_gathered"] = bool(gathered is not None and gathered.get("gathered"))
    manifest["n_system_shards"] = len(system_shards)
    weight_map: dict[str, str] = {}
    shard_blobs: list[tuple[str, bytes]] = []
    for i, shard in enumerate(system_shards):
        name = f"{SYSTEM_SHARD_PREFIX}{i:05d}{SYSTEM_SHARD_SUFFIX}"
        blob = safetensors_bytes(shard)
        shard_blobs.append((name, blob))
        for key in shard:
            weight_map[key] = name
    report = _parameter_report(system, gathered)
    index = {
        "weight_map": weight_map,
        "n_shards": len(system_shards),
        "parameters": report,
        "bool_fields": bool_fields,
    }
    index_blob = json.dumps(index, indent=2).encode("utf-8")
    sidecar = {
        "schema_version": 1,
        "scheduler": None,
        "rng": None,
        "optimizer_meta": opt_meta,
        "runtime_meta": runtime_meta,
        "has_optimizer": opt_state is not None,
        "has_optimizer_tensors": bool(opt_tensors),
        "has_runtime_tensors": bool(runtime_tensors),
    }
    if runtime_meta is not None:
        sidecar["scheduler"] = runtime_meta.get("scheduler")
        sidecar["rng"] = runtime_meta.get("rng")
        sidecar["runtime_meta"] = {k: v for k, v in runtime_meta.items() if k not in ("scheduler", "rng")}
    sidecar_blob = json.dumps(sidecar, indent=2).encode("utf-8")
    files: dict[str, bytes] = {}
    if opt_tensors:
        files[OPTIMIZER_NAME] = safetensors_bytes(opt_tensors)
    if runtime_tensors:
        files[RUNTIME_TENSORS_NAME] = safetensors_bytes(runtime_tensors)
    manifest["files"] = {
        name: sha256_bytes(blob) for name, blob in shard_blobs
    }
    manifest["files"][TENSORS_INDEX_NAME] = sha256_bytes(index_blob)
    manifest["files"][SIDECAR_NAME] = sha256_bytes(sidecar_blob)
    for name, blob in files.items():
        manifest["files"][name] = sha256_bytes(blob)
    # Tensor payloads are already dense binary blobs. Deflate burns CPU for
    # minutes on 6.8B checkpoints and stalls the H200 sanity run after step 10.
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(MANIFEST_NAME, yaml.safe_dump(manifest, sort_keys=False))
        zf.writestr(CONFIG_NAME, yaml.safe_dump(asdict(system.config), sort_keys=False))
        zf.writestr(IDENTITY_NAME, _identity_json(extras))
        zf.writestr(TENSORS_INDEX_NAME, index_blob)
        zf.writestr(SIDECAR_NAME, sidecar_blob)
        for name, blob in shard_blobs:
            zf.writestr(name, blob)
        for name, blob in files.items():
            zf.writestr(name, blob)
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
        names = zf.namelist()
        # 1. allowlist + legacy pickle reject, before any parsing.
        check_namelist(names)
        manifest = yaml.safe_load(zf.read(MANIFEST_NAME))
        _require(manifest, "format", "nullxes-minakanushi")
        _require(manifest, "architecture", "MINAKANUSHI")
        _require(manifest, "organization", "NULLXES")
        _require(manifest, "native_runtime", "nullxes")
        _require(manifest, "architecture_version", cfg.identity.architecture_version)
        if int(manifest.get("checkpoint_format_version", 0)) != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"checkpoint_format_version={manifest.get('checkpoint_format_version')!r} "
                f"requires v{CHECKPOINT_FORMAT_VERSION} (legacy pickle rejected)"
            )
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
        # 2. hash verify every payload file before tensor parsing.
        expected_files = manifest.get("files") or {}
        if not isinstance(expected_files, dict) or not expected_files:
            raise ValueError("checkpoint manifest has no files hash map")
        blobs: dict[str, bytes] = {}
        for name, digest in expected_files.items():
            if name in (MANIFEST_NAME, CONFIG_NAME, IDENTITY_NAME):
                continue
            if name not in names:
                raise ValueError(f"checkpoint manifest lists missing file {name!r}")
            data = zf.read(name)
            if sha256_bytes(data) != digest:
                raise ValueError(f"checkpoint file {name!r} hash mismatch (wrong_hash rejected)")
            blobs[name] = data
        # 3. schema parse (index + sidecar), no model mutation yet.
        index = json.loads(blobs[TENSORS_INDEX_NAME].decode("utf-8"))
        sidecar = json.loads(blobs[SIDECAR_NAME].decode("utf-8"))
        if not isinstance(sidecar, dict) or sidecar.get("schema_version") != 1:
            raise ValueError("sidecar.json schema_version must be 1")
        from minakanushi.training.mina_format import SIDECAR_ALLOWED_KEYS

        unknown_sidecar = set(sidecar) - SIDECAR_ALLOWED_KEYS
        if unknown_sidecar:
            raise ValueError(f"sidecar.json unexpected keys {sorted(unknown_sidecar)!r}")
        weight_map = index.get("weight_map") or {}
        shard_names = sorted({v for v in weight_map.values() if _SHARD_RE.match(v)})
        if not shard_names:
            raise ValueError("tensors_index.json has no system shards")
        # 4. load safetensors shards into memory (still no mutation).
        system_map: dict[str, torch.Tensor] = {}
        for name in shard_names:
            part = load_safetensors_bytes(blobs[name])
            for key, value in part.items():
                if key in system_map:
                    raise ValueError(f"duplicate system key {key!r} across shards")
                system_map[key] = value
        if set(system_map) != set(weight_map):
            raise ValueError("system keys do not match tensors_index weight_map")
        expected_state = system.state_dict()
        if set(system_map) != set(expected_state):
            missing = sorted(set(expected_state) - set(system_map))[:8]
            extra = sorted(set(system_map) - set(expected_state))[:8]
            raise ValueError(f"system key mismatch missing={missing!r} extra={extra!r}")
        for key in index.get("bool_fields") or []:
            if key in system_map:
                system_map[key] = system_map[key].to(dtype=torch.bool)
        for key, value in system_map.items():
            ref = expected_state[key]
            if tuple(value.shape) != tuple(ref.shape) or value.dtype != ref.dtype:
                raise ValueError(
                    f"system[{key!r}] shape/dtype {tuple(value.shape)}/{value.dtype} "
                    f"!= expected {tuple(ref.shape)}/{ref.dtype}"
                )
        opt_state = None
        if sidecar.get("has_optimizer"):
            if sidecar.get("has_optimizer_tensors"):
                if OPTIMIZER_NAME not in blobs:
                    raise ValueError("checkpoint declares optimizer but file is missing")
                opt_state = unflatten_optimizer(
                    load_safetensors_bytes(blobs[OPTIMIZER_NAME]),
                    sidecar.get("optimizer_meta") or {},
                )
            else:
                opt_state = unflatten_optimizer(
                    {},
                    sidecar.get("optimizer_meta") or {},
                )
        runtime = None
        if sidecar.get("has_runtime_tensors"):
            if RUNTIME_TENSORS_NAME not in blobs:
                raise ValueError("checkpoint declares runtime tensors but file is missing")
            runtime = decode_runtime(
                load_safetensors_bytes(blobs[RUNTIME_TENSORS_NAME]),
                (sidecar.get("runtime_meta") or {}),
            )
            # decode_runtime expects full meta incl. scheduler/rng; sidecar splits them.
            if runtime is not None:
                runtime["scheduler"] = sidecar.get("scheduler")
                # rng meta lives in sidecar; tensors already decoded — reattach via stored meta
                from minakanushi.training.mina_format import decode_rng

                runtime["rng"] = decode_rng(
                    sidecar.get("rng"),
                    load_safetensors_bytes(blobs[RUNTIME_TENSORS_NAME]),
                )
        elif sidecar.get("runtime_meta") is not None or sidecar.get("scheduler") is not None or sidecar.get("rng") is not None:
            # runtime meta without tensors (scheduler/rng only, e.g. trainer path)
            runtime = decode_runtime({}, {
                "scheduler": sidecar.get("scheduler"),
                "rng": sidecar.get("rng"),
                "world": None,
                "last_predicted": None,
                "plant": None,
            })
        payload = {
            "system": system_map,
            "parameter_report": (index.get("parameters") or {}),
            "optimizer": opt_state,
            "runtime": runtime,
        }
    # 5. only now mutate model/optimizer.
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
