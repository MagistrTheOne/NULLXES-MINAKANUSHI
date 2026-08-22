"""v0.3.1 baseline freeze does not construct 6.8B."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import torch
import yaml

from minakanushi.training.baseline import inspect_mina, write_baseline


def _write_mina(path: Path) -> None:
    system = {"weight": torch.randn(8, 8), "bias": torch.randn(8)}
    payload = {"system": system, "optimizer": {"exp_avg": torch.ones(4)}, "runtime": {"rng": "present"}}
    manifest = {
        "architecture": "MINAKANUSHI",
        "organization": "NULLXES",
        "native_runtime": "nullxes",
        "latent_dim": 8,
        "train": {"step": 128, "dataset_cursor": 128, "scheduler": {"step_num": 128}, "metrics": {"loss": 1.25}},
    }
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.yaml", yaml.safe_dump(manifest, sort_keys=False))
        buf = io.BytesIO()
        torch.save(payload, buf)
        zf.writestr("weights.pt", buf.getvalue())


def test_inspect_mina_hashes_and_lists_resume_keys(tmp_path: Path) -> None:
    mina = tmp_path / "minakanushi_stage0_step128.mina"
    _write_mina(mina)
    report = inspect_mina(mina)
    assert len(report["sha256"]) == 64
    assert report["research_scale"] is False
    assert report["step"] == 128
    assert report["resume_keys"]["optimizer"] is True
    assert report["resume_keys"]["scheduler"] is True
    assert report["resume_keys"]["dataset_cursor"] is True
    out = tmp_path / "anchor"
    written = write_baseline(mina, out)
    assert (out / "sha256.txt").read_text(encoding="utf-8").strip() == report["sha256"]
    metrics = json.loads((out / "metrics_before.json").read_text(encoding="utf-8"))
    assert metrics["sha256"] == written["sha256"]
    assert metrics["metrics"]["loss"] == 1.25
    assert "LlamaForCausalLM" not in json.dumps(metrics)
