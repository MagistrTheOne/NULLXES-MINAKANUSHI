"""MINA v3 checkpoint payload format: SAFETENSORS-INSIDE / PICKLE-OUT.

Layout inside ``*.mina`` (zip, STORED):

    manifest.yaml                  identity + dims + train extras + files sha256
    architecture.yaml              full ArchitectureConfig
    identity.json                  MINAKANUSHI identity bundle pointer
    weights/tensors_index.json     system shard map + parameter report
    weights/system-00000.safetensors ...
    weights/optimizer.safetensors  (optional, AdamW tensor state only)
    weights/runtime.safetensors    (optional, world/last_predicted/rng tensors)
    weights/sidecar.json           typed non-tensor state (schema_version=1)

No ``*.pt``, no ``torch.save``/``torch.load``, no ``weights_only=False``.
Any legacy pickle entry -> reject before any model mutation.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

MANIFEST_NAME = "manifest.yaml"
CONFIG_NAME = "architecture.yaml"
IDENTITY_NAME = "identity.json"
TENSORS_INDEX_NAME = "weights/tensors_index.json"
SIDECAR_NAME = "weights/sidecar.json"
OPTIMIZER_NAME = "weights/optimizer.safetensors"
RUNTIME_TENSORS_NAME = "weights/runtime.safetensors"
SYSTEM_SHARD_PREFIX = "weights/system-"
SYSTEM_SHARD_SUFFIX = ".safetensors"

CHECKPOINT_FORMAT_VERSION = 3
SIDECAR_SCHEMA_VERSION = 1

_TOP_LEVEL_ALLOW = frozenset({MANIFEST_NAME, CONFIG_NAME, IDENTITY_NAME})
_FIXED_ALLOW = frozenset({TENSORS_INDEX_NAME, SIDECAR_NAME, OPTIMIZER_NAME, RUNTIME_TENSORS_NAME})
_SHARD_RE = re.compile(r"^weights/system-(\d{5})\.safetensors$")
_LEGACY_RE = re.compile(r"(^|/)weights\.pt$|\.pt$|sidecar\.pt$")

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

SIDECAR_ALLOWED_KEYS = frozenset({
    "schema_version",
    "scheduler",
    "rng",
    "optimizer_meta",
    "runtime_meta",
    "has_optimizer",
    "has_optimizer_tensors",
    "has_runtime_tensors",
})


def _require_safetensors():
    try:
        from safetensors.torch import load as _load, save as _save
    except ImportError as exc:
        raise SystemExit("safetensors is required for .mina v3: pip install 'nullxes-minakanushi[hf]'") from exc
    return _save, _load


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_namelist(names: list[str]) -> None:
    for name in names:
        if not name or name.startswith("/") or ".." in Path(name).parts or "\\" in name:
            raise ValueError(f"checkpoint path traversal rejected: {name!r}")
        if name.endswith("/"):
            raise ValueError(f"checkpoint directory entry rejected: {name!r}")
    legacy = [n for n in names if n == "weights.pt" or _LEGACY_RE.search(n)]
    if legacy:
        raise ValueError(
            f"legacy pickle payload rejected (SAFETENSORS-INSIDE/PICKLE-OUT): {legacy[:8]!r}"
        )
    unknown = [
        n for n in names
        if n not in _TOP_LEVEL_ALLOW
        and n not in _FIXED_ALLOW
        and not _SHARD_RE.match(n)
    ]
    if unknown:
        raise ValueError(f"checkpoint unexpected files rejected: {unknown[:8]!r}")
    if MANIFEST_NAME not in names:
        raise ValueError("checkpoint missing manifest.yaml")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


# ----------------------------------------------------------------------------
# optimizer flatten / unflatten (tensors <-> safetensors, meta <-> json)
# ----------------------------------------------------------------------------

_MOMENT_OK = frozenset({"exp_avg", "exp_avg_sq", "max_exp_avg_sq", "step"})


def flatten_optimizer(opt_state: Any) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Split AdamW state into safetensors map + JSON meta. No pickle."""
    from minakanushi.training.optimizer_state import _maybe_unflatten_integer_optimizer

    if opt_state is None:
        return {}, {"schema_version": 1, "empty": True, "param_groups": [], "state_fields": {}}
    nested = _maybe_unflatten_integer_optimizer(opt_state)
    raw_state = nested.get("state") or {}
    tensors: dict[str, Tensor] = {}
    fields: dict[str, dict[str, Any]] = {}
    for idx, blob in raw_state.items():
        if not isinstance(blob, dict):
            raise ValueError(f"optimizer state[{idx}] must be a dict")
        for name, value in blob.items():
            if torch.is_tensor(value):
                key = f"state.{int(idx)}.{name}"
                t = value.detach().cpu().contiguous()
                tensors[key] = t
            else:
                fields.setdefault(str(int(idx)), {})[str(name)] = _to_jsonable(value)
    groups_meta: list[dict[str, Any]] = []
    for group in nested.get("param_groups") or []:
        if not isinstance(group, dict):
            raise ValueError("optimizer param_groups entry must be a dict")
        meta_group = {k: _to_jsonable(v) for k, v in group.items() if k != "params"}
        params = group.get("params", [])
        if isinstance(params, dict):
            params = [params[i] for i in range(len(params))]
        meta_group["params"] = [int(p) for p in list(params)]
        groups_meta.append(meta_group)
    meta = {
        "schema_version": 1,
        "empty": False,
        "param_groups": groups_meta,
        "state_fields": fields,
    }
    return tensors, meta


def unflatten_optimizer(tensors: dict[str, Tensor], meta: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(meta, dict) or meta.get("schema_version") != 1:
        raise ValueError("optimizer_meta schema_version must be 1")
    if meta.get("empty"):
        return None
    state: dict[int, dict[str, Any]] = {}
    for key, value in tensors.items():
        parts = str(key).split(".", 2)
        if len(parts) != 3 or parts[0] != "state":
            raise ValueError(f"optimizer shard unexpected key {key!r}")
        idx = int(parts[1])
        state.setdefault(idx, {})[parts[2]] = value
    for idx, blob in (meta.get("state_fields") or {}).items():
        if not isinstance(blob, dict):
            raise ValueError("optimizer_meta state_fields entry must be a dict")
        state.setdefault(int(idx), {}).update(blob)
    groups = meta.get("param_groups") or []
    if not isinstance(groups, list):
        raise ValueError("optimizer_meta param_groups must be a list")
    return {"state": state, "param_groups": [dict(g) for g in groups]}


# ----------------------------------------------------------------------------
# scheduler / rng encode (strict schema)
# ----------------------------------------------------------------------------

def encode_scheduler(obj: Any) -> dict[str, Any] | None:
    if obj is None:
        return None
    if not isinstance(obj, dict):
        raise ValueError("scheduler state must be a dict")
    out = {
        "step_num": int(obj["step_num"]),
        "warmup_steps": int(obj["warmup_steps"]),
        "base_lr": float(obj["base_lr"]),
    }
    if out["step_num"] < 0 or out["warmup_steps"] < 0 or not (out["base_lr"] > 0):
        raise ValueError(f"scheduler state out of range: {out!r}")
    return out


def decode_scheduler(obj: Any) -> dict[str, Any] | None:
    if obj is None:
        return None
    return encode_scheduler(obj)


def encode_rng(obj: Any) -> tuple[dict[str, Any] | None, dict[str, Tensor]]:
    """Split RNG state into JSON part + tensor part. Strict schema."""
    if obj is None:
        return None, {}
    if not isinstance(obj, dict):
        raise ValueError("rng state must be a dict")
    allowed = {"python", "numpy", "torch", "cuda"}
    unknown = set(obj) - allowed
    if unknown:
        raise ValueError(f"rng unexpected keys {sorted(unknown)!r}")
    tensors: dict[str, Tensor] = {}
    py = obj.get("python")
    if py is not None:
        if not (isinstance(py, (list, tuple)) and len(py) == 3):
            raise ValueError("rng.python must be random.getstate() triple")
        version, internal, gauss = py
        internal = list(internal)
        if int(version) != 3 or len(internal) != 625:
            raise ValueError("rng.python must be version 3 with 625 ints")
        py = {"version": 3, "state": [int(x) for x in internal],
              "gauss": None if gauss is None else float(gauss)}
    npy = obj.get("numpy")
    if npy is not None:
        if not (isinstance(npy, (list, tuple)) and len(npy) == 5):
            raise ValueError("rng.numpy must be np.random.get_state() 5-tuple")
        tag, arr, pos, has_gauss, cached = npy
        if str(tag) != "MT19937":
            raise ValueError(f"rng.numpy bit generator {tag!r} unsupported (only MT19937)")
        arr = np.asarray(arr, dtype=np.uint32).reshape(-1)
        if arr.size != 624:
            raise ValueError("rng.numpy state must have 624 uint32")
        npy = {"bit_generator": "MT19937", "state": [int(x) for x in arr.tolist()],
               "pos": int(pos), "has_gauss": int(has_gauss), "cached_gaussian": float(cached)}
    torch_state = obj.get("torch")
    if torch_state is not None:
        if not torch.is_tensor(torch_state):
            raise ValueError("rng.torch must be a Tensor")
        tensors["rng.torch"] = torch_state.detach().cpu().contiguous()
    cuda = obj.get("cuda")
    if cuda is not None:
        if not isinstance(cuda, (list, tuple)):
            raise ValueError("rng.cuda must be a list")
        for i, t in enumerate(cuda):
            if not torch.is_tensor(t):
                raise ValueError("rng.cuda entries must be Tensors")
            tensors[f"rng.cuda.{i}"] = t.detach().cpu().contiguous()
        cuda = {"count": len(cuda)}
    return {"python": py, "numpy": npy, "has_torch": torch_state is not None, "cuda": cuda}, tensors


def decode_rng(meta: Any, tensors: dict[str, Tensor]) -> dict[str, Any] | None:
    if meta is None:
        return None
    if not isinstance(meta, dict):
        raise ValueError("rng meta must be a dict")
    out: dict[str, Any] = {}
    py = meta.get("python")
    if py is not None:
        out["python"] = (int(py["version"]), tuple(int(x) for x in py["state"]), py["gauss"])
    npy = meta.get("numpy")
    if npy is not None:
        out["numpy"] = (
            str(npy["bit_generator"]),
            np.array(list(npy["state"]), dtype=np.uint32),
            int(npy["pos"]), int(npy["has_gauss"]), float(npy["cached_gaussian"]),
        )
    if meta.get("has_torch"):
        t = tensors.get("rng.torch")
        if t is None or not torch.is_tensor(t):
            raise ValueError("rng.torch tensor missing")
        out["torch"] = t
    cuda_meta = meta.get("cuda")
    if cuda_meta is not None:
        n = int(cuda_meta["count"])
        out["cuda"] = [tensors[f"rng.cuda.{i}"] for i in range(n)]
    return out


# ----------------------------------------------------------------------------
# runtime (world / plant) encode
# ----------------------------------------------------------------------------

def _body_to_json(body: Any) -> dict[str, Any]:
    return {
        "body_id": int(body["body_id"]),
        "kind": str(body["kind"]),
        "xy": [float(v) for v in np.asarray(body["xy"]).reshape(-1).tolist()],
        "vel": [float(v) for v in np.asarray(body["vel"]).reshape(-1).tolist()],
        "size": [float(v) for v in np.asarray(body["size"]).reshape(-1).tolist()],
        "accel": None if body.get("accel") is None else [float(v) for v in np.asarray(body["accel"]).reshape(-1).tolist()],
    }


def _body_from_json(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("plant body must be a dict")
    for key in ("body_id", "kind", "xy", "vel", "size"):
        if key not in raw:
            raise ValueError(f"plant body missing {key!r}")
    return {
        "body_id": int(raw["body_id"]),
        "kind": str(raw["kind"]),
        "xy": np.array([float(v) for v in raw["xy"]], dtype=np.float64),
        "vel": np.array([float(v) for v in raw["vel"]], dtype=np.float64),
        "size": np.array([float(v) for v in raw["size"]], dtype=np.float64),
        "accel": None if raw.get("accel") is None else np.array([float(v) for v in raw["accel"]], dtype=np.float64),
    }


def _to_storable(t: Tensor, *, name: str) -> Tensor:
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bool:
        return t.to(dtype=torch.uint8)
    return t


def encode_runtime(tensors: dict[str, Any] | None) -> tuple[dict[str, Tensor], dict[str, Any] | None]:
    """Split runtime tensors dict into safetensors map + JSON meta."""
    if not tensors:
        return {}, None
    if not isinstance(tensors, dict):
        raise ValueError("runtime tensors must be a dict")
    allowed = {"rng", "scheduler", "world", "last_predicted", "plant"}
    unknown = set(tensors) - allowed
    if unknown:
        raise ValueError(f"runtime unexpected keys {sorted(unknown)!r}")
    tmap: dict[str, Tensor] = {}
    rng_meta, rng_tensors = encode_rng(tensors.get("rng"))
    tmap.update(rng_tensors)
    scheduler_meta = encode_scheduler(tensors.get("scheduler"))
    meta: dict[str, Any] = {"scheduler": scheduler_meta, "rng": rng_meta}
    for key in ("world", "last_predicted"):
        blob = tensors.get(key)
        if blob is None:
            meta[key] = None
            continue
        if not isinstance(blob, dict) or "tensors" not in blob:
            raise ValueError(f"runtime {key!r} must be a world dump {{tensors, ...}}")
        inner = blob["tensors"]
        missing = [f for f in WORLD_TENSOR_FIELDS if f not in inner]
        if missing:
            raise ValueError(f"runtime {key!r} missing fields {missing!r}")
        for field in WORLD_TENSOR_FIELDS:
            value = inner[field]
            if not torch.is_tensor(value):
                raise ValueError(f"runtime {key}.{field} must be a Tensor")
            stored = _to_storable(value, name=f"{key}.{field}")
            tmap[f"{key}.{field}"] = stored
            if field == "occupied":
                meta.setdefault("bool_fields", []).append(f"{key}.{field}")
        meta[key] = {"self_index": int(blob.get("self_index", 0)), "provenance": str(blob.get("provenance", "checkpoint"))}
    plant = tensors.get("plant")
    if plant is None:
        meta["plant"] = None
    else:
        if not isinstance(plant, dict):
            raise ValueError("runtime plant must be a dict")
        meta["plant"] = {
            "t": float(plant["t"]),
            "last_intent": str(plant.get("last_intent", "SAFE_HOLD")),
            "agent": _body_to_json(plant["agent"]),
            "movers": [_body_to_json(b) for b in plant.get("movers", [])],
            "obstacles": [_body_to_json(b) for b in plant.get("obstacles", [])],
            "targets": [_body_to_json(b) for b in plant.get("targets", [])],
            "hidden_ids": [int(i) for i in plant.get("hidden_ids", ())],
            "removed_ids": [int(i) for i in plant.get("removed_ids", ())],
            "rng_state": _to_jsonable(plant.get("rng_state")),
        }
    return tmap, meta


def decode_runtime(tmap: dict[str, Tensor], meta: Any) -> dict[str, Any] | None:
    if meta is None:
        return None
    if not isinstance(meta, dict):
        raise ValueError("runtime meta must be a dict")
    out: dict[str, Any] = {
        "scheduler": decode_scheduler(meta.get("scheduler")),
        "rng": decode_rng(meta.get("rng"), tmap),
    }
    bool_fields = set(meta.get("bool_fields") or [])
    for key in ("world", "last_predicted"):
        blob = meta.get(key)
        if blob is None:
            out[key] = None
            continue
        inner = {}
        for field in WORLD_TENSOR_FIELDS:
            t = tmap.get(f"{key}.{field}")
            if t is None or not torch.is_tensor(t):
                raise ValueError(f"runtime tensor {key}.{field} missing")
            if f"{key}.{field}" in bool_fields:
                t = t.to(dtype=torch.bool)
            inner[field] = t
        out[key] = {"tensors": inner, "self_index": int(blob.get("self_index", 0)),
                    "provenance": str(blob.get("provenance", "checkpoint"))}
    plant = meta.get("plant")
    if plant is None:
        out["plant"] = None
    else:
        out["plant"] = {
            "t": float(plant["t"]),
            "last_intent": str(plant.get("last_intent", "SAFE_HOLD")),
            "agent": _body_from_json(plant["agent"]),
            "movers": [_body_from_json(b) for b in plant.get("movers", [])],
            "obstacles": [_body_from_json(b) for b in plant.get("obstacles", [])],
            "targets": [_body_from_json(b) for b in plant.get("targets", [])],
            "hidden_ids": [int(i) for i in plant.get("hidden_ids", ())],
            "removed_ids": [int(i) for i in plant.get("removed_ids", ())],
            "rng_state": plant.get("rng_state"),
        }
    return out


def dump_world_tensors(world) -> dict[str, Tensor]:
    from minakanushi.runtime.snapshot import WORLD_TENSOR_FIELDS as _FIELDS

    return {name: getattr(world, name).detach().cpu().contiguous() for name in _FIELDS}


def safetensors_bytes(tmap: dict[str, Tensor]) -> bytes:
    save, _ = _require_safetensors()
    for key, value in tmap.items():
        if not torch.is_tensor(value):
            raise ValueError(f"safetensors map[{key!r}] is {type(value).__name__}, not a Tensor")
        if value.dtype == torch.bool:
            raise ValueError(
                f"safetensors map[{key!r}] is bool which safetensors cannot store; "
                "convert occupancy masks before saving"
            )
    return save(tmap)


def load_safetensors_bytes(data: bytes) -> dict[str, Tensor]:
    _, load = _require_safetensors()
    out = load(data)
    if not isinstance(out, dict):
        raise ValueError("safetensors payload is not a tensor map")
    return out


def split_for_shards(tmap: dict[str, Tensor], max_bytes: int) -> list[dict[str, Tensor]]:
    from minakanushi.training.shard import split_tensor_map

    return split_tensor_map(tmap, max_bytes)
