"""Identity Initialization — stamp + validate + save. Not a train run.

Does not construct 6.8B. Does not compute identity_loss. Copies checkpoint
bytes and replaces identity.json / manifest.train identity extras.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import yaml

from minakanushi.identity.authority import AuthorityMode, AuthorityModel
from minakanushi.identity.constants import (
    ARCHITECTURE_ID,
    ARCHITECTURE_NAME,
    NATIVE_RUNTIME,
    ORGANIZATION,
    SHORT_NAME,
    SYSTEM_CLASS,
)
from minakanushi.identity.self_model import SelfModel

IDENTITY_NAME = "identity.json"
MANIFEST_NAME = "manifest.yaml"
AUTHORITY_MODES = tuple(mode.value for mode in AuthorityMode)


class IdentityInitError(ValueError):
    """Fail-loud passport / authority / ActionIntent contract violation."""


def canonical_identity_payload(self_model: SelfModel | None = None, authority: AuthorityModel | None = None) -> dict:
    model = self_model or SelfModel()
    gate = authority or AuthorityModel()
    ident = model.identity
    return {
        "architecture": ARCHITECTURE_NAME,
        "short_name": SHORT_NAME,
        "architecture_id": ARCHITECTURE_ID,
        "organization": ORGANIZATION,
        "system_class": SYSTEM_CLASS,
        "native_runtime": NATIVE_RUNTIME,
        "identity_state": {
            "self_model": model.to_dict(),
            "authority": gate.to_dict(),
            "initialized": True,
            "trainable": False,
        },
        "identity_block": {
            "architecture_name": ident.architecture_name,
            "short_name": ident.short_name,
            "architecture_id": ident.architecture_id,
            "organization": ident.organization,
            "system_class": ident.system_class,
            "native_runtime": ident.native_runtime,
        },
        "embodiment": {
            "actuators": list(model.embodiment.actuators),
            "limitations": list(model.embodiment.limitations),
            "pwm": False,
        },
        "authority_schema": {
            "modes": list(AUTHORITY_MODES),
            "gate_after": "ActionIntent",
            "neural_objective": False,
        },
    }


def validate_identity_payload(payload: dict) -> None:
    if payload.get("architecture") != ARCHITECTURE_NAME:
        raise IdentityInitError(f"architecture={payload.get('architecture')!r}")
    if payload.get("short_name") != SHORT_NAME:
        raise IdentityInitError(f"short_name={payload.get('short_name')!r}")
    if payload.get("architecture_id") != ARCHITECTURE_ID:
        raise IdentityInitError(f"architecture_id={payload.get('architecture_id')!r}")
    if payload.get("organization") != ORGANIZATION:
        raise IdentityInitError(f"organization={payload.get('organization')!r}")
    if payload.get("native_runtime") != NATIVE_RUNTIME:
        raise IdentityInitError(f"native_runtime={payload.get('native_runtime')!r}")
    state = payload.get("identity_state") or {}
    if state.get("trainable") is True:
        raise IdentityInitError("identity must not be marked trainable")
    if "identity_loss" in payload or "identity_loss" in state:
        raise IdentityInitError("identity_loss is forbidden")
    emb = payload.get("embodiment") or {}
    actuators = tuple(emb.get("actuators") or ())
    if "intent_only" not in actuators:
        raise IdentityInitError("ActionIntent contract requires actuators intent_only")
    if emb.get("pwm") is True:
        raise IdentityInitError("pwm actuators are forbidden")
    schema = payload.get("authority_schema") or {}
    modes = tuple(schema.get("modes") or ())
    for required in AUTHORITY_MODES:
        if required not in modes:
            raise IdentityInitError(f"authority schema missing {required}")
    if schema.get("neural_objective") is True:
        raise IdentityInitError("authority is a gate, not a neural objective")
    self_raw = (state.get("self_model") or {}).get("identity") or {}
    if self_raw and self_raw.get("architecture_name") not in (None, ARCHITECTURE_NAME):
        raise IdentityInitError("SelfModel identity mismatch")


def _stamp_manifest(raw: bytes) -> bytes:
    manifest = yaml.safe_load(raw) or {}
    if manifest.get("architecture") != ARCHITECTURE_NAME:
        raise IdentityInitError("manifest architecture is not MINAKANUSHI")
    if manifest.get("organization") != ORGANIZATION:
        raise IdentityInitError("manifest organization is not NULLXES")
    if manifest.get("native_runtime") != NATIVE_RUNTIME:
        raise IdentityInitError("manifest native_runtime is not nullxes")
    train = dict(manifest.get("train") or {})
    train["identity_initialized"] = True
    train["identity_trainable"] = False
    manifest["train"] = train
    modules = dict(manifest.get("modules") or {})
    modules["self_model"] = True
    modules["authority"] = True
    manifest["modules"] = modules
    return yaml.safe_dump(manifest, sort_keys=False).encode("utf-8")


def initialize_identity(src: str | Path, dest: str | Path) -> Path:
    """Copy *.mina weights unchanged. Stamp identity passport. Fail loud."""
    src = Path(src)
    dest = Path(dest)
    if src.suffix != ".mina":
        raise IdentityInitError(f"checkpoint must be .mina, got {src}")
    if dest.suffix != ".mina":
        raise IdentityInitError(f"output must be .mina, got {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_identity_payload()
    validate_identity_payload(payload)
    identity_bytes = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED) as zout:
        names = set(zin.namelist())
        if MANIFEST_NAME not in names:
            raise IdentityInitError("checkpoint missing manifest.yaml")
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == IDENTITY_NAME:
                data = identity_bytes
            elif info.filename == MANIFEST_NAME:
                data = _stamp_manifest(data)
            zout.writestr(info, data)
        if IDENTITY_NAME not in names:
            zout.writestr(IDENTITY_NAME, identity_bytes)
    validate_bound_checkpoint(dest)
    return dest


def validate_bound_checkpoint(path: str | Path) -> dict:
    path = Path(path)
    with zipfile.ZipFile(path, "r") as zf:
        payload = json.loads(zf.read(IDENTITY_NAME).decode("utf-8"))
        manifest = yaml.safe_load(zf.read(MANIFEST_NAME))
    validate_identity_payload(payload)
    train = manifest.get("train") or {}
    if train.get("identity_initialized") is not True:
        raise IdentityInitError("manifest missing identity_initialized")
    if train.get("identity_trainable") is True:
        raise IdentityInitError("identity marked trainable in manifest")
    return payload
