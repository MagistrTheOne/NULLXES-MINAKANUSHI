"""6.8B pre-train episode curriculum. Not tokens. Not a humanoid network."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json

from minakanushi.architecture.config import SimulationConfig
from simulations.synthetic_world.dataset import generate_episode
from simulations.synthetic_world.dataset_v1 import episode_to_record
from simulations.synthetic_world.replay import canonical_json

PHASES: dict[str, tuple[str, ...]] = {
    "physics": ("const_velocity", "accelerate", "brake", "turn", "occlusion", "delayed", "obstacles"),
    "agency": ("const_velocity", "agent_move", "goal_change", "unexpected_stop"),
    "causality": ("hidden_correction", "conflict", "reacquisition", "gone_forever"),
    "embodiment": ("delayed", "motor_delay", "agent_move"),
}

PHASE_ORDER: tuple[str, ...] = ("physics", "agency", "causality", "embodiment")

EMBODIMENT = {
    "platform_type": "synthetic_agent",
    "height_cm": 175,
    "mass_kg": 72.0,
    "workspace": "synthetic_arena",
    "balance": "not_applicable_synthetic",
    "actuators": ("intent_only",),
    "pwm": False,
}


def _transitions(record: dict[str, Any]) -> list[dict[str, Any]]:
    observations = record["observations"]
    actions = record["actions"]
    belief = record["belief_states"]
    world = record["world_states"]
    outcomes = record.get("outcomes") or []
    rows: list[dict[str, Any]] = []
    for t in range(len(observations) - 1):
        frame_outcomes = outcomes[t] if t < len(outcomes) else []
        correction = next((row for row in frame_outcomes if row.get("type") == "correction"), None)
        lesson = "consistent"
        if correction is not None:
            lesson = str(correction.get("lesson", "transition_mismatch"))
        elif actions[t]["objective"] == "MOVE_TO" and world[t + 1]["xy"] != world[t]["xy"]:
            lesson = "I acted; world changed"
        rows.append(
            {
                "t": t,
                "observation_t": observations[t],
                "belief_t": belief[t],
                "action_t": actions[t],
                "world_t": world[t],
                "observation_t1": observations[t + 1],
                "correction": correction,
                "lesson": lesson,
            }
        )
    return rows


def episode_record(
    config: SimulationConfig,
    *,
    phase: str,
    seed: int,
    episode_index: int,
    length: int = 12,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"unknown curriculum phase {phase}")
    scenarios = PHASES[phase]
    scenario = scenarios[episode_index % len(scenarios)]
    episode = generate_episode(
        config,
        seed=seed,
        episode_index=episode_index,
        length=length,
        scenario=scenario,
    )
    record = episode_to_record(episode)
    record["phase"] = phase
    record["curriculum"] = "mina_6_8b"
    record["transitions"] = _transitions(record)
    record["embodiment"] = dict(EMBODIMENT)
    record["self_state"] = {
        **record.get("self_state", {}),
        "embodiment": dict(EMBODIMENT),
        "intent_only": True,
    }
    return record


def write_phase(
    root: Path,
    phase: str,
    config: SimulationConfig,
    *,
    seed: int,
    n_episodes: int,
    length: int = 12,
) -> list[Path]:
    dest = root / phase
    dest.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(n_episodes):
        record = episode_record(config, phase=phase, seed=seed, episode_index=i, length=length)
        path = dest / f"{record['episode_id']}.json"
        path.write_text(canonical_json(record), encoding="utf-8")
        paths.append(path)
    return paths


def write_index(root: Path, written: dict[str, list[Path]]) -> Path:
    root = Path(root)
    lines = []
    for phase in PHASE_ORDER:
        for path in written[phase]:
            rel = path.relative_to(root).as_posix()
            episode_id = path.stem
            scenario = episode_id.rsplit("-", 2)[0]
            lines.append(
                json.dumps(
                    {"path": rel, "phase": phase, "episode_id": episode_id, "scenario": scenario},
                    sort_keys=True,
                )
            )
    index = root / "index.jsonl"
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index


def write_curriculum(
    root: Path,
    config: SimulationConfig,
    *,
    seed: int,
    n_episodes: int,
    length: int = 12,
) -> dict[str, list[Path]]:
    root.mkdir(parents=True, exist_ok=True)
    written = {
        phase: write_phase(root, phase, config, seed=seed, n_episodes=n_episodes, length=length)
        for phase in PHASE_ORDER
    }
    write_index(root, written)
    return written
