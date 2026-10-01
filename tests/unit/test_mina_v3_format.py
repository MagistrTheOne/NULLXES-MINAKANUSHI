"""MINA v3 format: SAFETENSORS-INSIDE / PICKLE-OUT negative gates.

All rejects must happen BEFORE MODEL STATE MUTATION.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minakanushi.architecture.model import MinakanushiSystem
from minakanushi.training.checkpoint import load_mina, save_mina
from minakanushi.training.mina_format import (
    MANIFEST_NAME,
    check_namelist,
    decode_runtime,
    encode_runtime,
)

import helpers


def _small_system():
    cfg = helpers.cpu_config()
    return cfg, MinakanushiSystem(cfg.architecture)


def _snapshot(system):
    return {k: v.detach().clone() for k, v in system.state_dict().items()}


def _assert_unchanged(system, snap):
    for key, value in snap.items():
        assert torch.equal(value, system.state_dict()[key]), key


def test_v3_roundtrip_no_pickle(tmp_path: Path) -> None:
    cfg, system = _small_system()
    opt = torch.optim.AdamW(system.parameters(), lr=1e-3)
    before = _snapshot(system)
    path = tmp_path / "v3_step1.mina"
    save_mina(path, system, optimizer=opt, extras={"step": 1, "dataset_cursor": 1})
    names = zipfile.ZipFile(path).namelist()
    assert not any(n.endswith(".pt") for n in names)
    assert "weights/tensors_index.json" in names
    assert "weights/sidecar.json" in names
    manifest = yaml.safe_load(zipfile.ZipFile(path).read(MANIFEST_NAME))
    assert manifest["checkpoint_format_version"] == 3
    fresh_cfg, fresh = _small_system()
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    manifest2, payload = load_mina(path, fresh, optimizer=fresh_opt, return_payload=True)
    assert payload["optimizer"] is not None
    for key, value in before.items():
        assert torch.equal(value, fresh.state_dict()[key]), key


def _copy_with(names_values: dict[str, bytes], mutate) -> Path:
    raise AssertionError("use tmp_path caller")


def _rewrite(path: Path, out: Path, mutate) -> None:
    with zipfile.ZipFile(path) as zin:
        blobs = {n: zin.read(n) for n in zin.namelist()}
    blobs = mutate(dict(blobs))
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zout:
        for name, data in blobs.items():
            zout.writestr(name, data)


def _valid_mina(tmp_path: Path) -> Path:
    cfg, system = _small_system()
    opt = torch.optim.AdamW(system.parameters(), lr=1e-3)
    path = tmp_path / "valid.mina"
    save_mina(path, system, optimizer=opt, extras={"step": 1, "dataset_cursor": 1})
    return path


def test_legacy_pickle_rejected_before_mutation(tmp_path: Path) -> None:
    src = _valid_mina(tmp_path)
    out = tmp_path / "legacy_pickle.mina"

    def add_legacy(blobs):
        import torch as _t
        import io as _io
        buf = _io.BytesIO()
        _t.save({"evil": 1}, buf)
        blobs["weights.pt"] = buf.getvalue()
        return blobs

    _rewrite(src, out, add_legacy)
    _, fresh = _small_system()
    snap = _snapshot(fresh)
    with pytest.raises(ValueError, match="legacy pickle"):
        load_mina(out, fresh)
    _assert_unchanged(fresh, snap)


def test_wrong_hash_rejected_before_mutation(tmp_path: Path) -> None:
    src = _valid_mina(tmp_path)
    out = tmp_path / "wrong_hash.mina"

    def tamper(blobs):
        shard = next(n for n in blobs if n.startswith("weights/system-"))
        raw = bytearray(blobs[shard])
        raw[-8] ^= 0xFF
        blobs[shard] = bytes(raw)
        return blobs

    _rewrite(src, out, tamper)
    _, fresh = _small_system()
    snap = _snapshot(fresh)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_mina(out, fresh)
    _assert_unchanged(fresh, snap)


def test_unexpected_file_rejected(tmp_path: Path) -> None:
    src = _valid_mina(tmp_path)
    out = tmp_path / "unexpected_file.mina"

    def add(blobs):
        blobs["weights/evil.bin"] = b"nope"
        return blobs

    _rewrite(src, out, add)
    _, fresh = _small_system()
    with pytest.raises(ValueError, match="unexpected files"):
        load_mina(out, fresh)


def test_malformed_index_rejected(tmp_path: Path) -> None:
    src = _valid_mina(tmp_path)
    out = tmp_path / "malformed_index.mina"

    def break_index(blobs):
        from minakanushi.training.mina_format import TENSORS_INDEX_NAME
        index = json.loads(blobs[TENSORS_INDEX_NAME].decode())
        first = next(iter(index["weight_map"]))
        index["weight_map"][first] = "weights/system-99999.safetensors"
        blobs[TENSORS_INDEX_NAME] = json.dumps(index).encode()
        return blobs

    _rewrite(src, out, break_index)
    _, fresh = _small_system()
    with pytest.raises(ValueError):
        load_mina(out, fresh)


def test_wrong_shape_rejected_against_config(tmp_path: Path) -> None:
    _, system = _small_system()
    cfg2, _ = _small_system()
    src = _valid_mina(tmp_path)
    object.__setattr__(cfg2.architecture, "latent_dim", cfg2.architecture.latent_dim + 8)
    other = MinakanushiSystem(cfg2.architecture)
    with pytest.raises(ValueError, match="vs system"):
        load_mina(src, other)


def test_path_traversal_zip_rejected() -> None:
    with pytest.raises(ValueError, match="traversal"):
        check_namelist([MANIFEST_NAME, "../evil.safetensors"])
    with pytest.raises(ValueError, match="traversal"):
        check_namelist([MANIFEST_NAME, "/abs.safetensors"])


def test_runtime_encode_decode_world() -> None:
    import torch as _t

    cfg = helpers.cpu_config()
    from minakanushi.state.constructor import empty_world_state

    world = empty_world_state(cfg.architecture, 1, device=_t.device("cpu"), dtype=_t.float32)
    from minakanushi.runtime.snapshot import dump_world

    blob = dump_world(world)
    tmap, meta = encode_runtime({"world": blob, "last_predicted": None, "plant": None})
    assert any(k.startswith("world.") for k in tmap)
    assert meta["last_predicted"] is None
    back = decode_runtime(tmap, meta)
    assert back["world"] is not None
    assert back["last_predicted"] is None
    for field in ("entity_xy", "occupied", "existence"):
        assert _t.equal(back["world"]["tensors"][field], blob["tensors"][field]), field
