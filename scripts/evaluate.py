"""Evaluate architectural properties on SyntheticWorld. Do not run unless authorized."""

from __future__ import annotations

import argparse
from pathlib import Path

from minakanushi.architecture.config import load_config
from minakanushi.runtime.engine import MinakanushiEngine
from simulations.synthetic_world.world import SyntheticWorld


def evaluate(root: Path, steps: int = 32) -> dict[str, float]:
    config = load_config(
        root / "configs" / "architecture" / "cpu_dev.yaml",
        runtime_path=root / "configs" / "runtime" / "cpu.yaml",
        simulation_path=root / "configs" / "simulation" / "milestone1.yaml",
    )
    engine = MinakanushiEngine(config)
    world = SyntheticWorld(config.simulation, seed=config.runtime.seed)
    state = engine.initialize()
    rejected = 0
    entities = []
    for _ in range(steps):
        obs = world.observe()
        result = engine.step(obs, state)
        state = result.state
        world.step(result.action_intent)
        rejected += result.telemetry.rejected_strategies
        entities.append(result.telemetry.entity_count)
    return {
        "steps": float(steps),
        "mean_entities": float(sum(entities) / max(len(entities), 1)),
        "rejected_total": float(rejected),
        "final_uncertainty": float(result.telemetry.uncertainty),
        "parameters": float(engine.system.parameter_report()["trainable"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    metrics = evaluate(Path(__file__).resolve().parents[1], steps=args.steps)
    print(metrics)


if __name__ == "__main__":
    main()
