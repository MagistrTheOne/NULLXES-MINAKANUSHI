"""NULLXES SyntheticWorld Dataset v1 — episode contract and CPU writer.

Does not download data. Does not emit millions of samples.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from minakanushi.architecture.config import SimulationConfig
from simulations.synthetic_world.dataset import Episode, generate_episode
from simulations.synthetic_world.replay import canonical_json

SPLITS: tuple[str, ...] = ("train", "validation", "composition", "ood", "counterfactual")

SPLIT_SCENARIOS: dict[str, tuple[str, ...]] = {
    "train": ("const_velocity", "accelerate", "turn", "agent_move"),
    "validation": ("occlusion", "noisy", "obstacles"),
    "composition": ("accelerate", "turn", "occlusion", "agent_move"),
    "ood": ("gone_forever", "conflict", "hidden_correction"),
    "counterfactual": ("agent_move", "const_velocity"),
}


def _events_for_frame(episode: Episode, t: int) -> list[dict]:
    truth = episode.truth[t]
    prev = episode.truth[t - 1] if t > 0 else None
    events: list[dict] = []
    if truth.occluded_ids:
        events.append({"frame": t, "type": "occlusion", "ids": [int(i) for i in truth.occluded_ids]})
    if prev is not None:
        lost = set(prev.entity_id) - set(truth.entity_id)
        gained = set(truth.entity_id) - set(prev.entity_id)
        if lost:
            events.append({"frame": t, "type": "disappearance", "ids": sorted(int(i) for i in lost)})
        if gained:
            events.append({"frame": t, "type": "appearance", "ids": sorted(int(i) for i in gained)})
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
                "visible_ids": [int(v["id"]) for v in obs.visible],
                "occluded_ids": [int(i) for i in obs.occluded_ids],
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
