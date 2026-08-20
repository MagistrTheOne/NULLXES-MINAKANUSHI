"""v0.2 acceptance gate on cpu_dev. Does not construct 6.8B."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from helpers import ROOT


def _gate():
    path = ROOT / "scripts" / "gate_v02_acceptance.py"
    spec = importlib.util.spec_from_file_location("gate_v02_acceptance", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_v02_acceptance_gate_cpu_dev(tmp_path: Path) -> None:
    report = _gate().run_gate(tmp_path)
    assert report["predict_world"] is True
    assert report["detect_wrong_belief"] is True
    assert report["revise"] is True
    assert report["remember"] is True
    assert report["different_future"] is True
    assert report["respect_authority"] is True
    assert report["constraint_rejects_zone"] is True
    assert report["identity_bound"] is True
    assert report["pass"] is True
