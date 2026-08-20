"""Closed-loop SyntheticWorld demo. Do not run unless authorized."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minakanushi.architecture.config import load_config
from minakanushi.policy.intent import ActionIntent
from minakanushi.runtime.loop import MinakanushiRuntime
from simulations.synthetic_world.world import SyntheticWorld


def _intent_record(intent: ActionIntent) -> dict:
    return {
        "strategy_id": intent.strategy_id,
        "objective": intent.objective,
        "target_state": list(intent.target_state),
        "confidence": float(intent.confidence),
        "provenance": intent.provenance,
    }


def simulate(root: Path, steps: int) -> dict:
    config = load_config(
        root / "configs" / "architecture" / "cpu_dev.yaml",
        runtime_path=root / "configs" / "runtime" / "cpu.yaml",
        simulation_path=root / "configs" / "simulation" / "milestone1.yaml",
    )
    runtime = MinakanushiRuntime(config)
    cycles = []
    agent_path = []
    for _ in range(steps):
        before = tuple(float(x) for x in runtime.platform.world.agent.xy)
        result = runtime.cycle()
        tel = result.telemetry
        rec = {
            "cycle_id": tel.cycle_id,
            "physical_time": tel.physical_time,
            "world_entities": tel.entity_count,
            "active_events": tel.event_count,
            "memory_writes": tel.memory_writes,
            "memory_reads": tel.memory_reads,
            "mean_uncertainty": tel.uncertainty,
            "future_branches": tel.future_branches,
            "candidate_strategies": tel.candidate_strategies,
            "rejected_strategies": tel.rejected_strategies,
            "selected_strategy": tel.selected_strategy,
            "ActionIntent": _intent_record(result.action_intent),
            "agent_xy_before": list(before),
            "agent_xy_after_step": [float(x) for x in runtime.platform.world.agent.xy],
            "observation_count": tel.observation_count,
            "latency_ms": tel.latency_ms,
            "runtime_mode": result.runtime.mode,
        }
        cycles.append(rec)
        agent_path.append(tuple(float(x) for x in runtime.platform.world.agent.xy))
        print(json.dumps(rec, ensure_ascii=True), flush=True)

    # Proof: ActionIntent is applied to the world. Same seed, forced MOVE_TO
    # must diverge from a WAIT-only control if the step() path is live.
    ctrl = SyntheticWorld(config.simulation, seed=config.runtime.seed)
    moved = SyntheticWorld(config.simulation, seed=config.runtime.seed)
    wait = ActionIntent("wait", "WAIT", (1.0, 1.0), {}, 1.0, 1e9, (), "gate02.control")
    go = ActionIntent("move", "MOVE_TO", tuple(float(x) for x in config.simulation.targets[0]["xy"]), {}, 1.0, 1e9, (), "gate02.force")
    for _ in range(steps):
        ctrl.step(wait)
        moved.step(go)
    delta = float(((ctrl.agent.xy - moved.agent.xy) ** 2).sum() ** 0.5)
    live_delta = 0.0
    if len(agent_path) >= 2:
        live_delta = ((agent_path[-1][0] - agent_path[0][0]) ** 2 + (agent_path[-1][1] - agent_path[0][1]) ** 2) ** 0.5
    summary = {
        "steps": steps,
        "intent_affects_world": delta > 1e-6,
        "forced_move_vs_wait_agent_delta": delta,
        "live_agent_path_delta": live_delta,
        "mean_entities": sum(c["world_entities"] for c in cycles) / max(len(cycles), 1),
        "mean_uncertainty": sum(c["mean_uncertainty"] for c in cycles) / max(len(cycles), 1),
        "mean_memory_writes": sum(c["memory_writes"] for c in cycles) / max(len(cycles), 1),
        "mean_memory_reads": sum(c["memory_reads"] for c in cycles) / max(len(cycles), 1),
        "mean_future_branches": sum(c["future_branches"] for c in cycles) / max(len(cycles), 1),
        "mean_candidates": sum(c["candidate_strategies"] for c in cycles) / max(len(cycles), 1),
        "total_rejected": sum(c["rejected_strategies"] for c in cycles),
        "strategies": sorted({c["selected_strategy"] for c in cycles}),
    }
    print("SUMMARY " + json.dumps(summary, ensure_ascii=True), flush=True)
    if not summary["intent_affects_world"]:
        raise RuntimeError("ActionIntent did not affect later world state")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()
    simulate(Path(__file__).resolve().parents[1], args.steps)


if __name__ == "__main__":
    main()
