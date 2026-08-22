"""CPU stats for the v0.3.1 verdict. Does not construct 6.8B."""

from __future__ import annotations

import torch

from minakanushi.training.parallel import dense_system_state
from minakanushi.training.v031_verdict import compare_reports, summarize


def test_dense_system_state_leaves_ordinary_tensors() -> None:
    weight = torch.ones(2, 3)
    out = dense_system_state({"w": weight, "n": 1})
    assert out["w"] is weight
    assert out["n"] == 1


def test_summarize_mean_median_p90_worst10() -> None:
    stats = summarize([1.0, 2.0, 3.0, 4.0, 10.0])
    assert stats["n"] == 5
    assert stats["mean"] == 4.0
    assert stats["median"] == 3.0
    assert stats["p90"] == 10.0
    assert stats["worst10"] == 10.0


def test_compare_reports_marks_b_when_heldout_moves_and_memory_fails() -> None:
    def _agg(ade: float, rev: float, false: float, direction: float) -> dict:
        return {
            "aggregates": {
                "future_ADE": {"mean": ade},
                "revision_detected": {"mean": rev},
                "false_revision_rate": {"mean": false},
                "revision_direction_accuracy": {"mean": direction},
            },
            "memory": {"pass": False},
            "counterfactual": {"pass_existence": False, "pass_diversity": False},
        }

    report = compare_reports(_agg(1.2, 0.1, 0.0, 0.0), _agg(0.4, 0.4, 0.02, 0.0))
    assert report["variant"] == "B"
    assert report["accepted"] is False
    assert "memory usefulness not demonstrated" in report["c_signals"]
    assert report["heldout_ADE"]["improved"] is True
