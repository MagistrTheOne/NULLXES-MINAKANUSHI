"""NULLXES SyntheticWorld Dataset v1 — episode contract and CPU writer.

Does not download data. Does not emit millions of samples.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from minakanushi.architecture.config import SimulationConfig
from minakanushi.perception.bridge import Observation
from simulations.synthetic_world.dataset import Episode, FrameTruth, generate_episode
from simulations.synthetic_world.replay import canonical_json

SPLITS: tuple[str, ...] = ("train", "validation", "composition", "ood", "counterfactual")

SPLIT_SCENARIOS: dict[str, tuple[str, ...]] = {
    "train": (
        "const_velocity",
        "accelerate",
        "turn",
        "agent_move",
        "hidden_correction",
        "conflict",
        "reacquisition",
        "gone_forever",
    ),
    "validation": ("occlusion", "noisy", "obstacles"),
    "composition": ("accelerate", "turn", "occlusion", "agent_move"),
    "ood": ("gone_forever", "conflict", "hidden_correction"),
    "counterfactual": ("agent_move", "const_velocity"),
}


def _xy_of(truth, eid: int) -> np.ndarray | None:
    for i, item in enumerate(truth.entity_id):
        if int(item) == int(eid):
            return np.asarray(truth.xy[i], dtype=np.float64)
    return None


def _events_for_frame(episode: Episode, t: int) -> list[dict]:
    truth = episode.truth[t]
    prev = episode.truth[t - 1] if t > 0 else None
    events: list[dict] = []
    agent_xy = _xy_of(truth, 1)
    occlusion_ids: list[int] = []
    out_of_range_ids: list[int] = []
    if agent_xy is not None:
        for eid in truth.occluded_ids:
            pos = _xy_of(truth, int(eid))
            if pos is None:
                continue
            dist = float(np.linalg.norm(pos - agent_xy))
            if dist > float(episode.sensor_range) + 1e-9:
                out_of_range_ids.append(int(eid))
            else:
                occlusion_ids.append(int(eid))
    if occlusion_ids:
        events.append({"frame": t, "type": "occlusion", "ids": occlusion_ids})
    if out_of_range_ids:
        events.append({"frame": t, "type": "out_of_range", "ids": out_of_range_ids})
    if prev is not None:
        lost = set(int(i) for i in prev.entity_id) - set(int(i) for i in truth.entity_id)
        gained = set(int(i) for i in truth.entity_id) - set(int(i) for i in prev.entity_id)
        if lost:
            events.append({"frame": t, "type": "disappearance", "ids": sorted(lost)})
        if gained:
            events.append({"frame": t, "type": "appearance", "ids": sorted(gained)})
        if episode.scenario == "conflict" and t >= 4:
            events.append({"frame": t, "type": "conflict", "ids": [int(i) for i in truth.entity_id if i != 1]})
        if episode.scenario == "turn" and t == len(episode.truth) // 2:
            events.append({"frame": t, "type": "turn", "ids": [int(i) for i in truth.entity_id if i != 1]})
    return events


def _outcomes_for_frame(episode: Episode, t: int) -> list[dict]:
    if t + 1 >= len(episode.truth):
        return []
    cur = episode.truth[t]
    nxt = episode.truth[t + 1]
    nxt_xy = {int(eid): nxt.xy[i] for i, eid in enumerate(nxt.entity_id)}
    rows = []
    for eid, path in cur.future_xy.items():
        actual = nxt_xy.get(int(eid))
        if actual is None or path is None or len(path) == 0:
            continue
        pred = np.asarray(path[0], dtype=np.float64)
        err = float(np.linalg.norm(pred - np.asarray(actual, dtype=np.float64)))
        lesson = "consistent"
        if err >= 0.25:
            lesson = "velocity hypothesis revised" if episode.scenario in {"turn", "conflict", "hidden_correction"} else "transition_mismatch"
        rows.append(
            {
                "frame": t,
                "entity_id": int(eid),
                "predicted": pred.tolist(),
                "actual": np.asarray(actual, dtype=np.float64).tolist(),
                "error": err,
                "lesson": lesson,
                "type": "correction" if err >= 0.25 else "ok",
            }
        )
    return rows


def episode_to_record(episode: Episode) -> dict:
    observations = []
    world_states = []
    actions = []
    future_branches = []
    belief_states: list[dict] = []
    events: list[dict] = []
    corrections: list[dict] = []
    for t, (obs, truth) in enumerate(zip(episode.observations, episode.truth, strict=True)):
        observations.append(
            {
                "timestamp": float(obs.timestamp),
                "agent_xy": [float(obs.agent_xy[0]), float(obs.agent_xy[1])],
                "agent_vel": [float(obs.agent_vel[0]), float(obs.agent_vel[1])],
                "heading": float(obs.heading),
                "health": float(obs.health),
                "battery": float(obs.battery),
                "visible_ids": [int(v["id"]) for v in obs.visible],
                "occluded_ids": [int(i) for i in obs.occluded_ids],
                "visible": [
                    {
                        "id": int(v["id"]),
                        "kind": str(v.get("kind", "unknown")),
                        "xy": [float(v["xy"][0]), float(v["xy"][1])],
                        "vel": [float(v.get("vel", (0.0, 0.0))[0]), float(v.get("vel", (0.0, 0.0))[1])],
                        "confidence": float(v.get("confidence", 1.0)),
                    }
                    for v in obs.visible
                ],
                "noise_std": float(obs.noise_std),
                "arrival_time": float(obs.arrival_time if obs.arrival_time is not None else obs.timestamp),
            }
        )
        world_states.append(
            {
                "event_time": float(truth.event_time),
                "entity_id": [int(i) for i in truth.entity_id],
                "kind": list(truth.kind),
                "xy": np.asarray(truth.xy, dtype=np.float64).tolist(),
                "vel": np.asarray(truth.vel, dtype=np.float64).tolist(),
            }
        )
        actions.append({"objective": str(truth.action), "target": [float(truth.action_target[0]), float(truth.action_target[1])]})
        future_branches.append({str(int(eid)): np.asarray(path, dtype=np.float64).tolist() for eid, path in truth.future_xy.items()})
        visible = set(int(v["id"]) for v in obs.visible)
        occluded = set(int(i) for i in obs.occluded_ids)
        belief_states.append(
            {
                "entities": [
                    {
                        "id": int(eid),
                        "kind": str(truth.kind[i]),
                        "confidence": 0.9 if int(eid) in visible or int(eid) == 1 else (0.35 if int(eid) in occluded else 0.2),
                    }
                    for i, eid in enumerate(truth.entity_id)
                ]
            }
        )
        events.extend(_events_for_frame(episode, t))
        for row in _outcomes_for_frame(episode, t):
            if row["type"] == "correction":
                corrections.append(row)
    return {
        "episode_id": f"{episode.scenario}-{episode.seed}-{episode.episode_index}",
        "seed": int(episode.seed),
        "scenario": episode.scenario,
        "episode_index": int(episode.episode_index),
        "dt": float(episode.dt),
        "sensor_range": float(episode.sensor_range),
        "self_state": {"platform": "synthetic", "kind": "agent"},
        "observations": observations,
        "world_states": world_states,
        "belief_states": belief_states,
        "actions": actions,
        "future_branches": future_branches,
        "events": events,
        "corrections": corrections,
        "outcomes": [_outcomes_for_frame(episode, t) for t in range(len(episode.truth))],
    }


def write_split(
    root: Path,
    split: str,
    config: SimulationConfig,
    *,
    seed: int,
    n_episodes: int,
    length: int = 8,
) -> list[Path]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split}")
    dest = root / split
    dest.mkdir(parents=True, exist_ok=True)
    scenarios = SPLIT_SCENARIOS[split]
    paths: list[Path] = []
    for i in range(n_episodes):
        scenario = scenarios[i % len(scenarios)]
        episode = generate_episode(config, seed=seed, episode_index=i, length=length, scenario=scenario)
        record = episode_to_record(episode)
        path = dest / f"{record['episode_id']}.json"
        path.write_text(canonical_json(record), encoding="utf-8")
        paths.append(path)
    return paths


REQUIRED_KEYS = (
    "episode_id",
    "seed",
    "scenario",
    "observations",
    "world_states",
    "belief_states",
    "actions",
    "events",
    "corrections",
)

REQUIRED_6_8B_KEYS = REQUIRED_KEYS + ("phase", "curriculum", "transitions", "embodiment")


def validate_episode_record(record: dict, *, curriculum_6_8b: bool = False) -> None:
    required = REQUIRED_6_8B_KEYS if curriculum_6_8b else REQUIRED_KEYS
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"episode missing keys {missing}")
    if len(record["observations"]) < 2:
        raise ValueError("episode needs at least two observations")
    if len(record["observations"]) != len(record["world_states"]):
        raise ValueError("observations / world_states length mismatch")
    if len(record["actions"]) != len(record["observations"]):
        raise ValueError("actions / observations length mismatch")
    if curriculum_6_8b and record.get("embodiment", {}).get("pwm") is True:
        raise ValueError("embodiment.pwm must be false")


def _visible_from_world(world: dict, visible_ids: list[int], obs_visible: list[dict] | None) -> tuple[dict, ...]:
    if obs_visible:
        rows = []
        for item in obs_visible:
            rows.append(
                {
                    "id": int(item["id"]),
                    "kind": str(item.get("kind", "unknown")),
                    "xy": (float(item["xy"][0]), float(item["xy"][1])),
                    "vel": (float(item.get("vel", (0.0, 0.0))[0]), float(item.get("vel", (0.0, 0.0))[1])),
                    "confidence": float(item.get("confidence", 1.0)),
                }
            )
        return tuple(rows)
    id_to_i = {int(eid): i for i, eid in enumerate(world["entity_id"])}
    rows = []
    for eid in visible_ids:
        i = id_to_i.get(int(eid))
        if i is None:
            continue
        xy = world["xy"][i]
        vel = world["vel"][i]
        rows.append(
            {
                "id": int(eid),
                "kind": str(world["kind"][i]),
                "xy": (float(xy[0]), float(xy[1])),
                "vel": (float(vel[0]), float(vel[1])),
                "confidence": 0.9,
            }
        )
    return tuple(rows)


def record_to_episode(record: dict, *, curriculum_6_8b: bool = False) -> Episode:
    validate_episode_record(record, curriculum_6_8b=curriculum_6_8b)
    observations: list[Observation] = []
    truth: list[FrameTruth] = []
    n = len(record["observations"])
    for t, obs_raw in enumerate(record["observations"]):
        world = record["world_states"][t]
        action = record["actions"][t]
        visible_ids = [int(i) for i in obs_raw.get("visible_ids", [])]
        occluded_ids = tuple(int(i) for i in obs_raw.get("occluded_ids", []))
        visible = _visible_from_world(world, visible_ids, obs_raw.get("visible"))
        agent_xy = (float(obs_raw["agent_xy"][0]), float(obs_raw["agent_xy"][1]))
        agent_vel = (float(obs_raw["agent_vel"][0]), float(obs_raw["agent_vel"][1]))
        arrival = obs_raw.get("arrival_time", obs_raw["timestamp"])
        observations.append(
            Observation(
                timestamp=float(obs_raw["timestamp"]),
                agent_xy=agent_xy,
                agent_vel=agent_vel,
                heading=float(obs_raw.get("heading", 0.0)),
                health=float(obs_raw.get("health", 1.0)),
                battery=float(obs_raw.get("battery", 1.0)),
                visible=visible,
                occluded_ids=occluded_ids,
                noise_std=float(obs_raw.get("noise_std", 0.0)),
                arrival_time=float(arrival),
                metadata={"event_time": float(world.get("event_time", obs_raw["timestamp"])), "scenario": str(record["scenario"])},
            )
        )
        future_raw = record["future_branches"][t] if t < len(record.get("future_branches") or []) else {}
        future_xy = {
            int(eid): np.asarray(path, dtype=np.float64)
            for eid, path in dict(future_raw).items()
        }
        entity_id = tuple(int(i) for i in world["entity_id"])
        truth.append(
            FrameTruth(
                entity_id=entity_id,
                kind=tuple(str(k) for k in world["kind"]),
                xy=np.asarray(world["xy"], dtype=np.float64),
                vel=np.asarray(world["vel"], dtype=np.float64),
                event_time=float(world.get("event_time", obs_raw["timestamp"])),
                arrival_time=float(arrival),
                visible_ids=tuple(visible_ids),
                occluded_ids=occluded_ids,
                noise_std=float(obs_raw.get("noise_std", 0.0)),
                action=str(action["objective"]),
                action_target=(float(action["target"][0]), float(action["target"][1])),
                future_xy=future_xy,
            )
        )
    return Episode(
        seed=int(record["seed"]),
        scenario=str(record["scenario"]),
        episode_index=int(record.get("episode_index", 0)),
        observations=observations,
        truth=truth,
        dt=float(record.get("dt", 0.1)),
        sensor_range=float(record.get("sensor_range", 4.0)),
    )
