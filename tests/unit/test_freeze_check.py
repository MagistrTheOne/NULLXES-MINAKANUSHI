"""Architecture freeze check does not construct 6.8B."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import torch
import yaml

from minakanushi.training.freeze_check import check_mina_freeze, check_yaml_freeze


def test_yaml_matches_7aba976() -> None:
    report = check_yaml_freeze()
    assert report["pass"] is True
    assert report["constructed_6_8b"] is False
    assert report["layers"] == 32
    assert report["contract_hash"] == report["want_hash"]


def test_probe_mina_is_not_the_frozen_6_8b(tmp_path: Path) -> None:
    mina = tmp_path / "probe.mina"
    payload = {"system": {"w": torch.ones(2, 2)}}
    manifest = {"architecture": "MINAKANUSHI", "latent_dim": 8, "train": {"step": 1}}
    with zipfile.ZipFile(mina, "w") as zf:
        zf.writestr("manifest.yaml", yaml.safe_dump(manifest))
        buf = io.BytesIO()
        torch.save(payload, buf)
        zf.writestr("weights.pt", buf.getvalue())
    report = check_mina_freeze(mina)
    assert report["constructed_6_8b"] is False
    assert report["pass"] is False
    assert "latent_dim" in report["drift"]
