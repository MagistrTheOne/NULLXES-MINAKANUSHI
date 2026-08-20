"""Resume the same training run. Weights-only load is a clone, not a continuation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import Optimizer

from minakanushi.architecture.model import MinakanushiSystem
from minakanushi.training.checkpoint import load_mina
from minakanushi.utils.seed import restore_rng


@dataclass
class ResumeState:
    last_step: int
    dataset_cursor: int
    extras: dict


class WarmupScheduler:
    """Linear warmup then constant LR. Identity scheduler when warmup_steps=0."""

    def __init__(self, optimizer: Optimizer, warmup_steps: int, base_lr: float) -> None:
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.base_lr = float(base_lr)
        self.step_num = 0

    def step(self) -> None:
        self.step_num += 1
        if self.warmup_steps <= 0:
            return
        scale = min(1.0, self.step_num / float(self.warmup_steps))
        for group in self.optimizer.param_groups:
            group["lr"] = self.base_lr * scale

    def state_dict(self) -> dict:
        return {"step_num": self.step_num, "warmup_steps": self.warmup_steps, "base_lr": self.base_lr}

    def load_state_dict(self, state: dict) -> None:
        self.step_num = int(state.get("step_num", 0))
        self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))
        self.base_lr = float(state.get("base_lr", self.base_lr))


def apply_resume(
    path: str | Path,
    system: MinakanushiSystem,
    optimizer: Optimizer,
    scheduler: WarmupScheduler,
) -> ResumeState:
    manifest, payload = load_mina(path, system, optimizer=optimizer, return_payload=True)
    extras = dict(manifest.get("train") or {})
    last_step = int(extras.get("step", 0))
    cursor = int(extras.get("dataset_cursor", last_step))
    sched = extras.get("scheduler")
    if isinstance(sched, dict):
        scheduler.load_state_dict(sched)
    elif payload.get("runtime") and isinstance(payload["runtime"], dict) and "scheduler" in payload["runtime"]:
        scheduler.load_state_dict(payload["runtime"]["scheduler"])
    rng = extras.get("rng")
    if rng is None and isinstance(payload.get("runtime"), dict):
        rng = payload["runtime"].get("rng")
    restore_rng(rng)
    return ResumeState(last_step=last_step, dataset_cursor=cursor, extras=extras)
