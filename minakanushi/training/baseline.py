"""Inspect and freeze a *.mina baseline without constructing 6.8B.

Anchor files:

    sha256.txt
    metrics_before.json
    reference_inference_before.pt   (cpu_dev only; 6.8B inference is load_mina on H200)
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import yaml

from minakanushi.training.checkpoint import MANIFEST_NAME, WEIGHTS_NAME

CHUNK = 1024 * 1024
RESEARCH_LATENT = 4096


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def inspect_mina(path: str | Path) -> dict[str, Any]:
    """Read manifest + zip inventory. Does not load 6.8B tensors or construct the net."""
    path = Path(path)
    if path.suffix != ".mina":
        raise ValueError(f"baseline must be *.mina, got {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    names: list[str]
    manifest: dict[str, Any]
    with zipfile.ZipFile(path, "r") as zf:
        names = list(zf.namelist())
        if MANIFEST_NAME not in names:
            raise ValueError("checkpoint missing manifest.yaml")
        manifest = yaml.safe_load(zf.read(MANIFEST_NAME))
    if manifest.get("architecture") != "MINAKANUSHI":
        raise ValueError("refusing non-MINAKANUSHI checkpoint")
    train = dict(manifest.get("train") or {})
    sidecar = "weights/sidecar.pt" in names
    weights = WEIGHTS_NAME in names
    inventory = {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "architecture": manifest.get("architecture"),
        "latent_dim": int(manifest.get("latent_dim") or 0),
        "sharded": bool(manifest.get("sharded")),
        "members": names,
        "has_weights_pt": weights,
        "has_optimizer_sidecar": sidecar or weights,
        "optimizer_loaded": False,
        "step": int(train["step"]) if train.get("step") is not None else None,
        "dataset_cursor": int(train["dataset_cursor"]) if train.get("dataset_cursor") is not None else None,
        "dataset_root": train.get("dataset_root"),
        "dataset_name": train.get("dataset_name"),
        "scheduler": train.get("scheduler"),
        "seed": train.get("seed"),
        "metrics": train.get("metrics"),
        "loss": train.get("loss"),
        "identity_initialized": train.get("identity_initialized"),
        "resume_keys": {
            "optimizer": sidecar or weights,
            "scheduler": isinstance(train.get("scheduler"), dict),
            "dataset_cursor": train.get("dataset_cursor") is not None,
            "rng": sidecar or weights,
        },
    }
    inventory["research_scale"] = int(inventory["latent_dim"]) >= RESEARCH_LATENT
    return inventory


def write_baseline(path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Write sha256 + metrics_before.json. Never constructs minakanushi_6_8b."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory = inspect_mina(path)
    (out_dir / "sha256.txt").write_text(inventory["sha256"] + "\n", encoding="utf-8")
    (out_dir / "metrics_before.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    inventory["out_dir"] = str(out_dir)
    inventory["reference_inference"] = str(out_dir / "reference_inference_before.pt")
    return inventory
