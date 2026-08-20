"""Core v0.2 metrics: ADE/FDE/uncertainty, revision, memory_future_delta, futures."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from minakanushi.architecture.config import load_config
from minakanushi.training.trainer import Trainer

ROOT = Path(__file__).resolve().parents[2]

CORE = (
    "future_ADE",
    "future_FDE",
    "uncertainty_calibration_error",
    "revision_accuracy",
    "revision_latency",
    "false_revision_rate",
    "memory_future_delta",
    "future_diversity",
    "counterfactual_quality",
)


def test_core_metrics_present_on_cpu_dev(tmp_path: Path) -> None:
    cfg = load_config(
        ROOT / "configs" / "architecture" / "cpu_dev.yaml",
        training_path=ROOT / "configs" / "training" / "stage0_overfit.yaml",
        runtime_path=ROOT / "configs" / "runtime" / "cpu.yaml",
        simulation_path=ROOT / "configs" / "simulation" / "milestone1.yaml",
    )
    cfg = replace(
        cfg,
        training=replace(
            cfg.training,
            steps=1,
            eval_every=1,
            checkpoint_every=1,
            log_every=1,
            sequence_length=6,
        ),
    )
    trainer = Trainer(cfg, ROOT)
    logs = trainer.fit(tmp_path)
    assert logs[0].metrics is not None
    for key in CORE:
        assert key in logs[0].metrics
