"""Gate 03B diagnostic aggregates hidden_correction across episodes."""

from __future__ import annotations

import importlib.util

from helpers import ROOT
from minakanushi.training.trainer import trainer_from_files


def _load(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_unroll_accepts_forced_scenario() -> None:
    trainer = trainer_from_files(ROOT, ROOT / "configs" / "training" / "gate03_revision_validation.yaml")
    pkt = trainer.unroll(1, scenario="hidden_correction", episode_index=3)
    assert pkt.scenario == "hidden_correction"
    assert pkt.episode_index == 3
    assert pkt.frame_index == 6


def test_gate03b_aggregates_hidden_direction() -> None:
    g03b = _load("gate03b_hidden_direction.py")
    trainer = trainer_from_files(ROOT, ROOT / "configs" / "training" / "gate03_revision_validation.yaml")
    report = g03b.diagnose(trainer, n=2, seed0=0)
    assert report["architecture"] == "MINAKANUSHI"
    assert report["lambda_revision"] == 1.0
    hidden = report["classes"]["hidden_correction"]
    assert hidden["direction"]["n"] == 2.0
    assert "median" in hidden["direction"]
    assert "revision_latency" in hidden
    assert "state_over_revision" in hidden["terms"]
    assert report["classes"]["conflict"]["direction"]["n"] == 2.0
    assert report["classes"]["reacquisition"]["direction"]["n"] == 2.0
    assert "stuck_prior" in report["verdict"] or "capacity_or_seed" in report["verdict"]
