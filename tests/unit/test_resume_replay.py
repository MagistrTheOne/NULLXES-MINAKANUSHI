"""Resume must continue the same pupil, not clone weights into a fresh Adam."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from minakanushi.architecture.config import load_config
from minakanushi.training.trainer import Trainer

ROOT = Path(__file__).resolve().parents[2]


def _cpu_trainer(**training_kw) -> Trainer:
    cfg = load_config(
        ROOT / "configs" / "architecture" / "cpu_dev.yaml",
        training_path=ROOT / "configs" / "training" / "stage0_overfit.yaml",
        runtime_path=ROOT / "configs" / "runtime" / "cpu.yaml",
        simulation_path=ROOT / "configs" / "simulation" / "milestone1.yaml",
    )
    train = replace(cfg.training, steps=1, eval_every=1, checkpoint_every=1, log_every=1, **training_kw)
    return Trainer(replace(cfg, training=train), ROOT)


def _first_moment(opt) -> torch.Tensor:
    param = next(iter(opt.param_groups[0]["params"]))
    return opt.state[param]["exp_avg"].detach().clone()


def test_resume_replays_optimizer_scheduler_cursor(tmp_path: Path) -> None:
    first = _cpu_trainer()
    first.fit(tmp_path / "run")
    mina = tmp_path / "run" / "minakanushi_stage0_step1.mina"
    assert mina.is_file()
    moment = _first_moment(first.opt)
    sched = first.scheduler.step_num

    continued = _cpu_trainer()
    continued.resume_from(mina)
    assert continued.start_step == 2
    assert continued.dataset_cursor == 1
    assert continued.scheduler.step_num == sched
    assert torch.allclose(_first_moment(continued.opt), moment)

    again = _cpu_trainer()
    again.resume_from(mina)
    log_a = continued.step_once(2)
    log_b = again.step_once(2)
    assert abs(log_a.loss - log_b.loss) < 1e-5
    assert abs(log_a.grad_norm - log_b.grad_norm) < 1e-4

    clone = _cpu_trainer()
    clone.system.load_state_dict(first.system.state_dict())
    clone_param = next(iter(clone.opt.param_groups[0]["params"]))
    assert clone_param not in clone.opt.state
    clone.step_once(2)
    assert clone_param in clone.opt.state
    assert int(clone.opt.state[clone_param]["step"].item()) == 1
    continued_param = next(iter(continued.opt.param_groups[0]["params"]))
    assert int(continued.opt.state[continued_param]["step"].item()) >= 2
