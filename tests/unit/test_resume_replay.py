"""Resume continues the same run: optimizer + RNG + cursor, not a weight clone."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from minakanushi.architecture.config import load_config
from minakanushi.training.checkpoint import latest_mina
from minakanushi.training.trainer import Trainer

ROOT = Path(__file__).resolve().parents[2]


def _cfg(steps: int, checkpoint_every: int):
    cfg = load_config(
        ROOT / "configs" / "architecture" / "cpu_dev.yaml",
        training_path=ROOT / "configs" / "training" / "stage0_overfit.yaml",
        runtime_path=ROOT / "configs" / "runtime" / "cpu.yaml",
        simulation_path=ROOT / "configs" / "simulation" / "milestone1.yaml",
    )
    return replace(
        cfg,
        training=replace(
            cfg.training,
            steps=steps,
            checkpoint_every=checkpoint_every,
            eval_every=1000,
            log_every=1000,
            n_overfit_episodes=8,
            sequence_length=6,
        ),
    )


def test_resume_step11_matches_continuous(tmp_path: Path) -> None:
    continuous = Trainer(_cfg(11, 11), ROOT)
    cont_logs = continuous.fit(tmp_path / "cont")
    cont11 = next(x for x in cont_logs if x.step == 11)

    first = Trainer(_cfg(10, 10), ROOT)
    first.fit(tmp_path / "seg")
    ckpt = latest_mina(tmp_path / "seg")
    resumed = Trainer(_cfg(1, 1), ROOT)
    logs = resumed.fit(tmp_path / "resume", resume=ckpt)
    assert resumed.start_step == 11
    assert logs[0].step == 11
    assert abs(logs[0].loss - cont11.loss) < 1e-4
    assert abs(logs[0].grad_norm - cont11.grad_norm) < 1e-3
    assert resumed._resume_extras.get("identity_initialized") is True
    assert resumed._resume_extras.get("identity_trainable") is False
