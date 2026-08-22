"""6.8B episode curriculum. v0.3 = longer chains + denser revisions. Not tokens."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from minakanushi.architecture.config import SimulationConfig
from minakanushi.policy.intent import ActionIntent
from simulations.synthetic_world.dataset import generate_episode, revision_frame
from simulations.synthetic_world.dataset_v1 import episode_to_record
from simulations.synthetic_world.replay import canonical_json
from simulations.synthetic_world.world import SyntheticWorld

PHASES: dict[str, tuple[str, ...]] = {
    "physics": (
        "const_velocity",
        "accelerate",
        "brake",
        "turn",
        "wrong_velocity",
        "occlusion",
        "delayed",
        "obstacles",
    ),
    "agency": (
        "const_velocity",
        "agent_move",
        "goal_change",
        "unexpected_stop",
        "follow",
        "avoid",
    ),
    "causality": (
        "hidden_correction",
        "hidden_correction_l2",
        "hidden_correction_l3",
        "hidden_object",
        "conflict",
        "reacquisition",
        "gone_forever",
    ),
    "embodiment": ("delayed", "sensor_delay", "motor_delay", "agent_move"),
}

PHASE_ORDER: tuple[str, ...] = ("physics", "agency", "causality", "embodiment")

# v0.3 trajectory lengths. Not a DWC change.
PHASE_LENGTHS: dict[str, int] = {
    "physics": 32,
    "agency": 32,
    "causality": 64,
    "embodiment": 64,
}

CURRICULUM_NAME = "mina_6_8b_v03"

CORRECTION_TYPES = (
    "wrong_velocity",
    "wrong_intent",
    "hidden_object",
    "agent_changes_goal",
    "unexpected_physics",
    "sensor_delay",
)

FORK_STRATEGIES: tuple[str, ...] = ("WAIT", "MOVE_TO", "FOLLOW", "AVOID")

SCENARIO_CORRECTION_TYPE: dict[str, str] = {
    "turn": "wrong_velocity",
    "wrong_velocity": "wrong_velocity",
    "accelerate": "wrong_velocity",
    "brake": "unexpected_physics",
    "unexpected_stop": "unexpected_physics",
    "goal_change": "agent_changes_goal",
    "hidden_correction": "hidden_object",
    "hidden_correction_l1": "hidden_object",
    "hidden_correction_l2": "wrong_velocity",
    "hidden_correction_l3": "wrong_intent",
    "hidden_object": "hidden_object",
    "reacquisition": "hidden_object",
    "conflict": "wrong_intent",
    "gone_forever": "hidden_object",
    "delayed": "sensor_delay",
    "sensor_delay": "sensor_delay",
    "motor_delay": "sensor_delay",
    "occlusion": "hidden_object",
}

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
    """Slim transitions: trainer reads observations[], not these copies."""
    observations = record["observations"]
    actions = record["actions"]
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
                "observation_t": {
                    "timestamp": observations[t]["timestamp"],
                    "agent_xy": observations[t]["agent_xy"],
                },
                "belief_t": {"n_entities": len(record["belief_states"][t].get("entities") or [])},
                "action_t": actions[t],
                "observation_t1": {
                    "timestamp": observations[t + 1]["timestamp"],
                    "agent_xy": observations[t + 1]["agent_xy"],
                },
                "correction": correction,
                "lesson": lesson,
            }
        )
    return rows


def _typed_event_corrections(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Stamp typed revisions on frames that already change the world.

    Short Gate-03 lengths keep a single primary stamp. v0.3 32/64 adds the
    later evidence / second cause, not random noise.
    """
    scenario = str(record["scenario"])
    kind = SCENARIO_CORRECTION_TYPE.get(scenario)
    length = len(record["observations"])
    planned: list[tuple[int, str]] = []
    if kind is not None:
        planned.append((revision_frame(scenario, length), kind))
    if length > 12:
        mid = length // 2
        late = (length * 3) // 4
        if scenario in {"turn", "wrong_velocity"}:
            planned.append((late, "wrong_velocity"))
        elif scenario == "accelerate":
            planned.append((late, "unexpected_physics"))
        elif scenario in {"brake", "unexpected_stop"}:
            planned.append((late, "unexpected_physics"))
        elif scenario == "goal_change":
            planned.append((mid, "agent_changes_goal"))
            planned.append((late, "wrong_intent"))
        elif scenario in {"hidden_correction", "hidden_object", "reacquisition", "hidden_correction_l1"}:
            planned.append((1, "hidden_object"))
            planned.append((late, "wrong_intent"))
        elif scenario == "hidden_correction_l2":
            planned.append((1, "hidden_object"))
            planned.append((late, "wrong_velocity"))
        elif scenario == "hidden_correction_l3":
            planned.append((1, "hidden_object"))
            planned.append((late, "wrong_intent"))
        elif scenario == "conflict":
            planned.append((late, "wrong_intent"))
        elif scenario in {"delayed", "sensor_delay", "motor_delay"}:
            planned.append((2, "sensor_delay"))
            planned.append((mid, "sensor_delay"))
        elif scenario == "occlusion":
            planned.append((mid, "hidden_object"))
        elif scenario == "gone_forever":
            planned.append((late, "hidden_object"))
        elif scenario == "const_velocity":
            planned.append((late, "unexpected_physics"))
        elif scenario == "obstacles":
            planned.append((late, "unexpected_physics"))
        elif scenario == "agent_move":
            planned.append((mid, "agent_changes_goal"))
        elif scenario == "follow":
            planned.append((mid, "wrong_intent"))
        elif scenario == "avoid":
            planned.append((mid, "unexpected_physics"))
    seen: set[tuple[int, str]] = set()
    rows: list[dict[str, Any]] = []
    for frame, correction_type in planned:
        idx = int(max(0, min(frame, length - 2)))
        key = (idx, correction_type)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "frame": idx,
                "entity_id": 2,
                "type": "correction",
                "correction_type": correction_type,
                "lesson": correction_type,
                "error": 0.0,
                "predicted": None,
                "actual": None,
            }
        )
    return rows


def _tag_natural_corrections(record: dict[str, Any]) -> None:
    default = SCENARIO_CORRECTION_TYPE.get(str(record["scenario"]))
    if default is None:
        return
    for row in record.get("corrections") or []:
        if not row.get("correction_type"):
            row["correction_type"] = default


def _fork_intent(world: SyntheticWorld, strategy: str) -> ActionIntent:
    agent = (float(world.agent.xy[0]), float(world.agent.xy[1]))
    if strategy == "WAIT":
        target = agent
    elif strategy == "MOVE_TO":
        tgt = world.targets[0]
        target = (float(tgt.xy[0]), float(tgt.xy[1]))
    elif strategy == "FOLLOW":
        mover = world.movers[0]
        target = (float(mover.xy[0]), float(mover.xy[1]))
    else:
        wall = world.obstacles[0]
        away = world.agent.xy - wall.xy
        nrm = float(np.linalg.norm(away))
        step = away / nrm if nrm > 1e-6 else np.array([1.0, 0.0], dtype=np.float64)
        nxt = world.agent.xy + step
        target = (float(nxt[0]), float(nxt[1]))
    return ActionIntent(
        strategy_id=strategy.lower(),
        objective=strategy,
        target_state=target,
        parameters={},
        confidence=1.0,
        valid_until=1e9,
        abort_conditions=(),
        provenance="curriculum.v03.fork",
    )


def _counterfactuals(config: SimulationConfig, *, seed: int, episode_index: int, horizon: int = 4) -> dict[str, Any]:
    """Same world seed, same first observation, four ActionIntents → four terminals."""
    local_seed = int(seed) * 1_000_003 + int(episode_index) * 9176
    terminals: dict[str, list[float]] = {}
    paths: dict[str, list[list[float]]] = {}
    for strategy in FORK_STRATEGIES:
        world = SyntheticWorld(config, seed=local_seed)
        intent = _fork_intent(world, strategy)
        path = [world.agent.xy.astype(np.float64).tolist()]
        for _ in range(horizon):
            world.step(intent)
            if strategy == "FOLLOW":
                intent = _fork_intent(world, strategy)
            path.append(world.agent.xy.astype(np.float64).tolist())
        terminals[strategy] = path[-1]
        paths[strategy] = path
    pts = [np.asarray(terminals[name], dtype=np.float64) for name in FORK_STRATEGIES]
    diversity = 0.0
    for i, a in enumerate(pts):
        for b in pts[i + 1 :]:
            diversity = max(diversity, float(np.linalg.norm(a - b)))
    return {
        "same_observation": True,
        "strategies": list(FORK_STRATEGIES),
        "terminals": terminals,
        "paths": paths,
        "future_diversity": diversity,
    }


def episode_record(
    config: SimulationConfig,
    *,
    phase: str,
    seed: int,
    episode_index: int,
    length: int | None = None,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"unknown curriculum phase {phase}")
    if length is None:
        length = PHASE_LENGTHS[phase]
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
    extra = _typed_event_corrections(record)
    record["corrections"] = list(record.get("corrections") or []) + extra
    _tag_natural_corrections(record)
    forks = _counterfactuals(config, seed=seed, episode_index=episode_index)
    record["phase"] = phase
    record["curriculum"] = CURRICULUM_NAME
    record["transitions"] = _transitions(record)
    record["embodiment"] = dict(EMBODIMENT)
    record["counterfactuals"] = forks
    record["future_diversity"] = forks["future_diversity"]
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
    length: int | None = None,
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
    length: int | None = None,
    lengths: dict[str, int] | None = None,
) -> dict[str, list[Path]]:
    root.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[Path]] = {}
    for phase in PHASE_ORDER:
        phase_len = length if length is not None else (lengths or PHASE_LENGTHS)[phase]
        written[phase] = write_phase(
            root, phase, config, seed=seed, n_episodes=n_episodes, length=phase_len
        )
    write_index(root, written)
    from minakanushi.training.heldout import write_heldout_split

    write_heldout_split(root)
    return written
