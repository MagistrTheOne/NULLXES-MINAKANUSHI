"""Gate 03 exam runner exists and hits the three revision classes."""

from __future__ import annotations

import importlib.util

from helpers import ROOT
from minakanushi.training.trainer import trainer_from_files


def _gate03():
    path = ROOT / "scripts" / "gate03_revision_validate.py"
    spec = importlib.util.spec_from_file_location("gate03_revision_validate", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_gate03_exam_runs_three_classes() -> None:
    g03 = _gate03()
    trainer = trainer_from_files(ROOT, ROOT / "configs" / "training" / "gate03_revision_validation.yaml")
    report = g03.run_exam(trainer)
    names = [row["scenario"] for row in report["scenarios"]]
    assert names == list(g03.EXAM_SCENARIOS)
    hidden = report["scenarios"][0]
    assert hidden["scenario"] == "hidden_correction"
    assert hidden["frame_index"] == 6
    assert hidden["constructor_corrections"] >= 1
    reacq = report["scenarios"][2]
    assert reacq["scenario"] == "reacquisition"
    assert reacq["identity"] == "same_hypothesis_revised"
    mean = report["summary"]["mean"]
    assert "revision_detected" in mean
    assert "revision_direction_accuracy" in mean
    assert "false_revision_rate" in mean
    assert report["lambda_revision"] == 1.0
