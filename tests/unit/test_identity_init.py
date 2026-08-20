"""Identity Initialization stamps a passport. It does not train identity."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from minakanushi.architecture.freeze import V02_CYCLE, V02_FORBIDDEN
from minakanushi.architecture.model import MinakanushiSystem
from minakanushi.identity.constants import ARCHITECTURE_NAME, SHORT_NAME
from minakanushi.identity.initialize import (
    IdentityInitError,
    initialize_identity,
    validate_bound_checkpoint,
)
from minakanushi.training.checkpoint import save_mina
from helpers import cpu_config


def test_v02_forbids_identity_loss_and_new_layers() -> None:
    assert V02_CYCLE == "pipeline_and_data_only"
    assert "identity_loss" in V02_FORBIDDEN
    assert "adding layers" in V02_FORBIDDEN
    assert "replacing DWC" in V02_FORBIDDEN
    assert "training authority as a neural objective" in V02_FORBIDDEN


def test_identity_init_stamps_passport_without_loading_weights(tmp_path: Path) -> None:
    cfg = cpu_config()
    system = MinakanushiSystem(cfg.architecture)
    src = tmp_path / "step64.mina"
    save_mina(src, system, extras={"step": 64, "dataset_cursor": 64})
    dest = tmp_path / "MINA-6.8B-IdentityBound.mina"
    initialize_identity(src, dest)
    payload = validate_bound_checkpoint(dest)
    assert payload["architecture"] == ARCHITECTURE_NAME
    assert payload["short_name"] == SHORT_NAME
    assert payload["identity_state"]["trainable"] is False
    assert payload["embodiment"]["pwm"] is False
    assert "identity_loss" not in json.dumps(payload)
    with zipfile.ZipFile(src) as a, zipfile.ZipFile(dest) as b:
        src_weights = {n: a.read(n) for n in a.namelist() if n.startswith("weights") or n == "weights.pt"}
        dst_weights = {n: b.read(n) for n in b.namelist() if n.startswith("weights") or n == "weights.pt"}
    assert src_weights == dst_weights


def test_identity_init_rejects_corrupt_passport(tmp_path: Path) -> None:
    cfg = cpu_config()
    system = MinakanushiSystem(cfg.architecture)
    src = tmp_path / "step64.mina"
    save_mina(src, system, extras={"step": 64})
    dest = tmp_path / "bad.mina"
    initialize_identity(src, dest)
    corrupted = tmp_path / "corrupt.mina"
    with zipfile.ZipFile(dest, "r") as zin, zipfile.ZipFile(corrupted, "w", compression=zipfile.ZIP_STORED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == "identity.json":
                data = json.dumps({"architecture": "NOTMINA"}).encode("utf-8")
            zout.writestr(info, data)
    with pytest.raises(IdentityInitError):
        validate_bound_checkpoint(corrupted)
