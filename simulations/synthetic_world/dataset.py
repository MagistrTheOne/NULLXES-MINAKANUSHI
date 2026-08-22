"""Deterministic Stage-0 synthetic episode generator.

Reproduced by (seed, scenario, episode_index). Does not download data.
Does not emit millions of samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from minakanushi.architecture.config import SimulationConfig
from minakanushi.perception.bridge import Observation
from minakanushi.policy.intent import ActionIntent
from simulations.synthetic_world.world import SyntheticWorld


SCENARIOS: tuple[str, ...] = (
    "const_velocity",
    "accelerate",
    "turn",
    "occlusion",
    "noisy",
    "missing",
    "dual_rate",
    "delayed",
    "agent_move",
    "obstacles",
)

GATE03_SCENARIOS: tuple[str, ...] = (
    "hidden_correction",
    "conflict",
    "gone_forever",
    "accelerate",
    "turn",
    "occlusion",
    "delayed",
)

# Training mix. GATE03 cases are the textbook for "new evidence beats old hypothesis",
# not OOD leftovers. Tracking scenarios stay so false-revision has a negative class.
TRAIN_CURRICULUM: tuple[str, ...] = (
    "hidden_correction",
    "conflict",
    "reacquisition",
    "gone_forever",
    "const_velocity",
    "occlusion",
    "noisy",
    "accelerate",
)

# Observation index of the revision event at the original length-12 pack.
# Longer v0.3 trajectories scale this so history exists before evidence.
CORRECTION_FRAME: dict[str, int] = {
    "hidden_correction": 6,
    "hidden_correction_l1": 6,
    "hidden_correction_l2": 6,
    "hidden_correction_l3": 6,
    "reacquisition": 6,
    "hidden_object": 6,
    "conflict": 4,
    "gone_forever": 3,
}

PHYSICS_ALIAS: dict[str, str] = {
    "reacquisition": "hidden_correction",
    "hidden_object": "hidden_correction",
    "hidden_correction_l1": "hidden_correction",
    "hidden_correction_l2": "hidden_correction",
    "hidden_correction_l3": "hidden_correction",
    "wrong_velocity": "turn",
    "sensor_delay": "delayed",
}


def hidden_level(scenario: str) -> int:
    if str(scenario).endswith("_l3"):
        return 3
    if str(scenario).endswith("_l2"):
        return 2
    return 1


def revision_frame(scenario: str, length: int) -> int:
    """Frame whose transition carries the cognitive event.

    At length<=12 keep the original indices so Gate 03 tests stay bitwise.
    At 32/64 scale so the chain is observe → error → evidence → revision → new future.
    """
    if length < 2:
        raise ValueError("episode too short to train a transition")
    named = CORRECTION_FRAME.get(scenario)
    if named is None:
        idx = max(1, min(length // 2, length - 2))
    elif length <= 12:
        idx = min(named, length - 2)
    else:
        idx = int(round(named * (length - 1) / 11.0))
        idx = max(1, min(idx, length - 2))
    return int(max(0, min(idx, length - 2)))


def training_frame(scenario: str, length: int) -> int:
    """Select the transition that carries the cognitive event."""
    return revision_frame(scenario, length)


@dataclass
class FrameTruth:
    entity_id: tuple[int, ...]
    kind: tuple[str, ...]
    xy: np.ndarray
    vel: np.ndarray
    event_time: float
    arrival_time: float
    visible_ids: tuple[int, ...]
    occluded_ids: tuple[int, ...]
    noise_std: float
    action: str
    action_target: tuple[float, float]
    future_xy: dict[int, np.ndarray]


@dataclass
class Episode:
    seed: int
    scenario: str
    episode_index: int
    observations: list[Observation]
    truth: list[FrameTruth]
    dt: float
    sensor_range: float


def _intent(name: str, xy: tuple[float, float]) -> ActionIntent:
    return ActionIntent(
        strategy_id=name.lower(),
        objective=name,
        target_state=xy,
        parameters={},
        confidence=1.0,
        valid_until=1e9,
        abort_conditions=(),
        provenance="stage0.dataset",
    )


def _future_table(world: SyntheticWorld, horizon: int) -> dict[int, np.ndarray]:
    bodies = [world.agent, *world.movers, *world.obstacles, *world.targets]
    table = {}
    for body in bodies:
        if body.body_id in world.removed_ids:
            continue
        path = np.zeros((horizon, 2), dtype=np.float64)
        xy = body.xy.copy()
        vel = body.vel.copy()
        for h in range(horizon):
            xy = xy + vel * world.config.dt
            path[h] = xy
        table[body.body_id] = path
    return table


def generate_episode(
    config: SimulationConfig,
    *,
    seed: int,
    episode_index: int,
    length: int = 16,
    horizon: int = 4,
    scenario: str | None = None,
) -> Episode:
    name = scenario or SCENARIOS[episode_index % len(SCENARIOS)]
    physics = PHYSICS_ALIAS.get(name, name)
    local_seed = int(seed) * 1_000_003 + int(episode_index) * 9176
    world = SyntheticWorld(config, seed=local_seed)
    rng = np.random.default_rng(local_seed)
    rev = revision_frame(name, length)

    if physics == "accelerate":
        world.movers[0].accel = np.array([0.4, 0.0], dtype=np.float64)
        world.movers[0].vel = world.movers[0].vel * 0.2
    if physics == "brake":
        world.movers[0].accel = np.array([-0.5, 0.0], dtype=np.float64)
        world.movers[0].vel = np.array([0.9, 0.0], dtype=np.float64)
    if physics == "turn":
        world.movers[0].vel = np.array([0.6, 0.0])
    if physics == "hidden_correction":
        world.movers[0].xy = world.agent.xy + np.array([1.4, 0.0])
        world.movers[0].vel = np.array([-0.8, 0.0])
        world.hidden_ids = set()
    if physics == "conflict":
        world.movers[0].xy = world.agent.xy + np.array([1.2, 0.0])
        world.movers[0].vel = np.zeros(2)
    if physics == "gone_forever":
        world.movers[0].xy = world.agent.xy + np.array([1.3, 0.0])

    frames_obs: list[Observation] = []
    frames_gt: list[FrameTruth] = []
    delay = 0.15 if name in {"delayed", "sensor_delay"} else 0.0
    if name == "motor_delay":
        delay = 0.30

    for t in range(length):
        if physics == "accelerate":
            pass
        if physics == "turn" and t == length // 2:
            world.movers[0].vel = np.array([0.0, 0.7])
        if length > 12 and physics == "turn" and t == (length * 3) // 4:
            world.movers[0].vel = np.array([0.55, -0.35])
        if length > 12 and physics == "accelerate" and t == (length * 2) // 3:
            world.movers[0].accel = np.array([-0.6, 0.2], dtype=np.float64)
        if length > 12 and physics == "brake" and t == (length * 2) // 3:
            world.movers[0].accel = np.array([0.35, 0.15], dtype=np.float64)
            world.movers[0].vel = np.array([0.2, 0.4], dtype=np.float64)
        if physics == "unexpected_stop" and t == length // 2:
            world.movers[0].vel = np.zeros(2)
            world.movers[0].accel = None
        if length > 12 and physics == "unexpected_stop" and t == (length * 3) // 4:
            world.movers[0].vel = np.array([0.4, 0.3])
        if length > 12 and physics in {"const_velocity", "obstacles"} and t == (length * 3) // 4:
            world.movers[0].vel = np.array([float(world.movers[0].vel[0]), 0.5])
        if length > 12 and name == "follow" and t == length // 2:
            world.movers[0].vel = np.array([0.0, 0.6])
        if length > 12 and name == "avoid" and t == length // 2:
            world.obstacles[0].xy = world.agent.xy + np.array([0.85, 0.0])
        if length > 12 and name == "agent_move" and t == length // 2:
            alt = world.targets[min(1, len(world.targets) - 1)]
            world.targets[0].xy = alt.xy.copy()
        if physics == "hidden_correction":
            level = hidden_level(name)
            if 1 <= t < rev:
                world.hidden_ids.add(world.movers[0].body_id)
                if level == 2 and t == max(2, rev // 2):
                    world.movers[0].vel = np.array([0.0, 0.75], dtype=np.float64)
                if level == 3 and t == max(2, rev // 2):
                    world.movers[0].vel = np.array([0.7, 0.4], dtype=np.float64)
            elif t == rev:
                world.hidden_ids.discard(world.movers[0].body_id)
                if level <= 1:
                    world.movers[0].xy = world.agent.xy + np.array([1.4, 0.0])
                    world.movers[0].vel = np.zeros(2)
                elif level == 2:
                    world.movers[0].xy = world.agent.xy + np.array([1.4, 0.0])
                else:
                    world.movers[0].xy = world.agent.xy + np.array([0.2, 1.6])
                    world.movers[0].vel = np.array([-0.2, 0.8], dtype=np.float64)
            else:
                world.hidden_ids.discard(world.movers[0].body_id)
        if physics == "conflict":
            if t < rev:
                world.movers[0].xy = world.agent.xy + np.array([1.0, 0.0])
            else:
                world.movers[0].xy = world.agent.xy + np.array([3.0, 0.0])
                world.movers[0].vel = np.zeros(2)
        if physics == "gone_forever" and t >= rev:
            world.removed_ids.add(world.movers[0].body_id)
            world.hidden_ids.add(world.movers[0].body_id)
        if name == "motor_delay" and t < 2:
            intent = _intent("WAIT", (float(world.agent.xy[0]), float(world.agent.xy[1])))
        elif name == "goal_change":
            tgt = world.targets[0] if t < length // 2 else world.targets[min(1, len(world.targets) - 1)]
            intent = _intent("MOVE_TO", (float(tgt.xy[0]), float(tgt.xy[1])))
        elif name == "follow":
            mover = world.movers[0]
            intent = _intent("FOLLOW", (float(mover.xy[0]), float(mover.xy[1])))
        elif name == "avoid":
            wall = world.obstacles[0]
            away = world.agent.xy - wall.xy
            nrm = float(np.linalg.norm(away))
            step = away / nrm if nrm > 1e-6 else np.array([1.0, 0.0])
            tgt = world.agent.xy + step
            intent = _intent("AVOID", (float(tgt[0]), float(tgt[1])))
        elif name in {"agent_move", "motor_delay"}:
            intent = _intent("MOVE_TO", (float(world.targets[0].xy[0]), float(world.targets[0].xy[1])))
        else:
            intent = _intent("WAIT", (float(world.agent.xy[0]), float(world.agent.xy[1])))

        obs = world.observe()
        event_time = world.t
        arrival_time = event_time + delay
        if name == "dual_rate" and t % 2 == 1:
            obs = Observation(
                timestamp=event_time,
                agent_xy=obs.agent_xy,
                agent_vel=obs.agent_vel,
                heading=obs.heading,
                health=obs.health,
                battery=obs.battery,
                visible=(),
                occluded_ids=tuple(b.body_id for b in [*world.movers, *world.obstacles, *world.targets]),
                noise_std=obs.noise_std,
                arrival_time=arrival_time,
                source_rate_telemetry=20.0,
                source_rate_vector=5.0,
            )
        else:
            visible = list(obs.visible)
            if name == "missing" and t % 4 == 0 and visible:
                dropped = visible.pop(0)
                obs = Observation(
                    timestamp=event_time,
                    agent_xy=obs.agent_xy,
                    agent_vel=obs.agent_vel,
                    heading=obs.heading,
                    health=obs.health,
                    battery=obs.battery,
                    visible=tuple(visible),
                    occluded_ids=obs.occluded_ids + (int(dropped["id"]),),
                    noise_std=obs.noise_std,
                    arrival_time=arrival_time,
                    source_rate_telemetry=20.0,
                    source_rate_vector=10.0 if name != "dual_rate" else 5.0,
                )
            else:
                stamped = []
                for item in visible:
                    row = dict(item)
                    row["event_time"] = event_time
                    row["arrival_time"] = arrival_time
                    row["source_rate"] = 5.0 if name == "dual_rate" else 10.0
                    stamped.append(row)
                obs = Observation(
                    timestamp=event_time,
                    agent_xy=obs.agent_xy,
                    agent_vel=obs.agent_vel,
                    heading=obs.heading,
                    health=obs.health,
                    battery=obs.battery,
                    visible=tuple(stamped),
                    occluded_ids=obs.occluded_ids,
                    noise_std=world.config.sensor_noise_std if name == "noisy" else obs.noise_std,
                    arrival_time=arrival_time,
                    source_rate_telemetry=20.0,
                    source_rate_vector=5.0 if name == "dual_rate" else 10.0,
                    metadata={"event_time": event_time, "scenario": name},
                )

        gt_map = world.ground_truth()
        ids = tuple(gt_map.keys())
        kinds = tuple(str(gt_map[i]["kind"]) for i in ids)
        xy = np.array([gt_map[i]["xy"] for i in ids], dtype=np.float64)
        vel = np.array([gt_map[i]["vel"] for i in ids], dtype=np.float64)
        truth = FrameTruth(
            entity_id=ids,
            kind=kinds,
            xy=xy,
            vel=vel,
            event_time=event_time,
            arrival_time=arrival_time,
            visible_ids=tuple(int(v["id"]) for v in obs.visible),
            occluded_ids=obs.occluded_ids,
            noise_std=obs.noise_std,
            action=intent.objective,
            action_target=intent.target_state,
            future_xy=_future_table(world, horizon),
        )
        frames_obs.append(obs)
        frames_gt.append(truth)
        world.step(intent)
        if name == "occlusion":
            world.movers[0].xy = world.agent.xy + np.array([world.config.sensor_range + 1.0, 0.0])

        _ = rng.random()

    return Episode(
        seed=seed,
        scenario=name,
        episode_index=episode_index,
        observations=frames_obs,
        truth=frames_gt,
        dt=config.dt,
        sensor_range=float(config.sensor_range),
    )


def generate_overfit_set(config: SimulationConfig, *, seed: int, n_episodes: int, length: int) -> list[Episode]:
    if n_episodes < 8 or n_episodes > 32:
        raise ValueError("overfit set must be 8–32 episodes")
    return [
        generate_episode(
            config,
            seed=seed,
            episode_index=i,
            length=length,
            scenario=TRAIN_CURRICULUM[i % len(TRAIN_CURRICULUM)],
        )
        for i in range(n_episodes)
    ]
