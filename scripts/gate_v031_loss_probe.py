"""CPU stub: watch future / revision / memory losses and ADE(memory ON vs OFF).

cpu_dev only. Does not construct 6.8B.

    python scripts/gate_v031_loss_probe.py --steps 32
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from minakanushi.architecture.config import load_config
from minakanushi.training.trainer import Trainer

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--out", type=Path, default=Path("artifacts/v031/loss_probe.json"))
    args = parser.parse_args()
    cfg = load_config(
        ROOT / "configs" / "architecture" / "cpu_dev.yaml",
        training_path=ROOT / "configs" / "training" / "mina_v031_cpu_probe.yaml",
        runtime_path=ROOT / "configs" / "runtime" / "cpu.yaml",
        simulation_path=ROOT / "configs" / "simulation" / "milestone1.yaml",
    )
    cfg = replace(cfg, training=replace(cfg.training, steps=int(args.steps), eval_every=1, checkpoint_every=10_000))
    trainer = Trainer(cfg, ROOT)
    rows = []
    for step in range(1, int(args.steps) + 1):
        log = trainer.step_once(step)
        row = {
            "step": step,
            "loss": log.loss,
            "future": log.terms.get("future"),
            "revision": log.terms.get("revision"),
            "memory": log.terms.get("memory"),
            "grad_norm": log.grad_norm,
            "metrics": log.metrics,
        }
        rows.append(row)
        print(
            f"step={step} loss={log.loss:.4f} future={row['future']} "
            f"revision={row['revision']} memory={row['memory']}",
            flush=True,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    last = next((r for r in reversed(rows) if r.get("metrics")), rows[-1])
    metrics = last.get("metrics") or {}
    print(
        json.dumps(
            {
                "out": str(args.out),
                "memory_ade_on": metrics.get("memory_ade_on"),
                "memory_ade_off": metrics.get("memory_ade_off"),
                "memory_helps_future": metrics.get("memory_helps_future"),
                "memory_future_delta": metrics.get("memory_future_delta"),
                "memory_effect_delta": metrics.get("memory_effect_delta"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
