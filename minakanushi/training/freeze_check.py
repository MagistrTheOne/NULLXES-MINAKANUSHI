"""No architecture drift. Reads YAML / *.mina manifest. Does not construct 6.8B."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import yaml

from minakanushi.architecture.config import load_architecture
from minakanushi.architecture.freeze import (
    FROZEN_AT,
    FROZEN_CORE_DEPTH,
    FROZEN_LATENT_DIM,
    FROZEN_MEMORY_DIM,
    FROZEN_MEMORY_SLOTS,
    FROZEN_PARAM_ESTIMATE,
    FROZEN_STATE_DIM,
    FROZEN_WORLD_SLOTS,
    assert_6_8b_frozen,
)
from minakanushi.training.baseline import inspect_mina
from minakanushi.training.checkpoint import CONFIG_NAME, MANIFEST_NAME
from minakanushi.training.parameter_inventory import estimate_parameters

ROOT = Path(__file__).resolve().parents[2]
FROZEN_YAML = ROOT / "configs" / "architecture" / "minakanushi_6_8b.yaml"


def contract_hash(fields: dict[str, int]) -> str:
    blob = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def frozen_fields() -> dict[str, int]:
    return {
        "latent_dim": FROZEN_LATENT_DIM,
        "state_dim": FROZEN_STATE_DIM,
        "memory_dim": FROZEN_MEMORY_DIM,
        "world_slots": FROZEN_WORLD_SLOTS,
        "memory_slots": FROZEN_MEMORY_SLOTS,
        "core_depth": FROZEN_CORE_DEPTH,
        "parameters": FROZEN_PARAM_ESTIMATE,
    }


def check_yaml_freeze(path: str | Path | None = None) -> dict[str, Any]:
    cfg = load_architecture(path or FROZEN_YAML)
    assert_6_8b_frozen(cfg)
    fields = {
        "latent_dim": int(cfg.latent_dim),
        "state_dim": int(cfg.state_dim),
        "memory_dim": int(cfg.memory_dim),
        "world_slots": int(cfg.world_slots),
        "memory_slots": int(cfg.memory_slots),
        "core_depth": int(cfg.core_depth),
        "parameters": int(estimate_parameters(cfg)["total_estimate"]),
    }
    want = frozen_fields()
    drift = {key: {"got": fields[key], "want": want[key]} for key in want if fields[key] != want[key]}
    return {
        "source": str(path or FROZEN_YAML),
        "freeze": FROZEN_AT,
        "fields": fields,
        "contract_hash": contract_hash(fields),
        "want_hash": contract_hash(want),
        "layers": int(cfg.core_depth),
        "constructed_6_8b": False,
        "pass": not drift,
        "drift": drift,
    }


def check_mina_freeze(path: str | Path) -> dict[str, Any]:
    inventory = inspect_mina(path)
    with zipfile.ZipFile(path, "r") as zf:
        manifest = yaml.safe_load(zf.read(MANIFEST_NAME)) or {}
        arch_raw = {}
        if CONFIG_NAME in zf.namelist():
            arch_raw = yaml.safe_load(zf.read(CONFIG_NAME)) or {}
    merged = {**manifest, **arch_raw}
    fields = {
        "latent_dim": int(merged.get("latent_dim") or inventory.get("latent_dim") or 0),
        "state_dim": int(merged.get("state_dim") or merged.get("latent_dim") or 0),
        "memory_dim": int(merged.get("memory_dim") or merged.get("latent_dim") or 0),
        "world_slots": int(merged.get("world_slots") or 0),
        "memory_slots": int(merged.get("memory_slots") or 0),
        "core_depth": int(merged.get("core_depth") or 0),
        "parameters": int(merged.get("parameters") or 0),
    }
    want = frozen_fields()
    drift = {key: {"got": fields[key], "want": want[key]} for key in want if fields[key] != want[key]}
    if fields["latent_dim"] != FROZEN_LATENT_DIM:
        drift["latent_dim"] = {"got": fields["latent_dim"], "want": FROZEN_LATENT_DIM}
    return {
        "source": str(path),
        "freeze": FROZEN_AT,
        "sha256": inventory["sha256"],
        "step": inventory.get("step"),
        "fields": fields,
        "contract_hash": contract_hash(fields),
        "want_hash": contract_hash(want),
        "layers": fields["core_depth"],
        "constructed_6_8b": False,
        "pass": not drift,
        "drift": drift,
        "note": "contract hash is YAML/manifest dims, not DWC weight hash. Weights are checkpoint.sha256.",
    }
