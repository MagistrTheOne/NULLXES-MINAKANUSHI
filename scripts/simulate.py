"""Closed-loop SyntheticWorld demo. Do not run unless authorized."""

from __future__ import annotations

import argparse
from pathlib import Path

from minakanushi.architecture.config import load_config
from minakanushi.runtime.engine import MinakanushiEngine
from simulations.synthetic_world.world import SyntheticWorld


def simulate(root: Path, steps: int) -> None:
    config = load_config(
        root / "configs" / "architecture" / "cpu_dev.yaml",
        runtime_path=root / "configs" / "runtime" / "cpu.yaml",
        simulation_path=root / "configs" / "simulation" / "milestone1.yaml",
    )
    engine = MinakanushiEngine(config)
    world = SyntheticWorld(config.simulation, seed=config.runtime.seed)
    state = engine.initialize()
    for _ in range(steps):
        obs = world.observe()
        result = engine.step(obs, state)
        state = result.state
        world.step(result.action_intent)
        tel = result.telemetry
        print(
            f"t={tel.physical_time:.2f} obs={tel.observation_count} "
            f"ent={tel.entity_count} strat={tel.selected_strategy} "
            f"rej={tel.rejected_strategies} u={tel.uncertainty:.3f} "
            f"ms={tel.latency_ms:.1f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()
    simulate(Path(__file__).resolve().parents[1], args.steps)


if __name__ == "__main__":
    main()
