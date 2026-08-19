"""Curriculum over architecture-validation stages."""

from __future__ import annotations

from pathlib import Path

from minakanushi.architecture.config import TrainingConfig, load_training


STAGE_FILES = {
    0: "configs/training/stage0_validation.yaml",
    "0_overfit": "configs/training/stage0_overfit.yaml",
    1: "configs/training/stage1_world.yaml",
    2: "configs/training/stage2_temporal.yaml",
}


def load_stage(stage: int | str, root: str | Path = ".") -> TrainingConfig:
    if stage not in STAGE_FILES:
        raise ValueError(f"training stage {stage} is not in the active curriculum {tuple(STAGE_FILES)}")
    return load_training(Path(root) / STAGE_FILES[stage])
