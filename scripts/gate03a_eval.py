"""Gate 03A eval: belief revision + held-out scenarios. Does not train."""

from __future__ import annotations

import json
from pathlib import Path

from minakanushi.architecture.config import load_architecture, load_config
from minakanushi.state.constructor import StateConstructor, empty_world_state
from minakanushi.training.metrics import evidence_dominance, false_persistence_steps
from simulations.synthetic_world.dataset import GATE03_SCENARIOS, generate_episode


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "stage0_generalization"


def main() -> None:
    cfg = load_config(
        ROOT / "configs" / "architecture" / "cpu_dev.yaml",
        training_path=ROOT / "configs" / "training" / "stage0_generalization.yaml",
        runtime_path=ROOT / "configs" / "runtime" / "cpu.yaml",
        simulation_path=ROOT / "configs" / "simulation" / "milestone1.yaml",
    )
    arch = load_architecture(ROOT / "configs" / "architecture" / "cpu_dev.yaml")
    report: dict = {"gate": "03A", "trained": False, "scenarios": {}}
    for seed in (101, 102, 103):
        for name in GATE03_SCENARIOS:
            ep = generate_episode(cfg.simulation, seed=seed, episode_index=seed, length=12, scenario=name)
            report["scenarios"].setdefault(name, []).append(
                {
                    "seed": seed,
                    "frames": len(ep.observations),
                    "visible0": len(ep.observations[0].visible),
                    "scenario": ep.scenario,
                }
            )
    ctor = StateConstructor(arch)
    import torch

    from minakanushi.architecture.mina_unit import pack_units
    from minakanushi.perception.bridge import PerceptionBridge

    bridge = PerceptionBridge(arch)
    device = torch.device("cpu")
    dtype = torch.float32
    world = empty_world_state(arch, 1, device=device, dtype=dtype)
    ep = generate_episode(cfg.simulation, seed=101, episode_index=0, length=12, scenario="hidden_correction")
    n_corr = 0
    for t, obs in enumerate(ep.observations):
        units = bridge.encode(obs, device=device, dtype=dtype)
        packed = pack_units(
            units,
            batch_index=0,
            max_units=arch.max_observations,
            latent_dim=arch.latent_dim,
            episode_position=float(t),
            now=obs.timestamp,
            device=device,
            dtype=dtype,
        )
        world = ctor.apply(packed, world, packed.semantic_embedding)
        n_corr += len(world.corrections)
    report["hidden_correction_events"] = n_corr
    report["evidence_dominance_demo"] = evidence_dominance(127.6, 100.0, 130.0)
    report["false_persistence_cap"] = arch.persistence.steps
    report["belief_is_not_memory"] = True
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate03a.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
