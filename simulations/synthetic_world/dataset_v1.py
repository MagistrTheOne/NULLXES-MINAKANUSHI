"""NULLXES SyntheticWorld Dataset v1 — episode contract and CPU writer.

Does not download data. Does not emit millions of samples.
Belief/future fields may be empty until a closed-loop recorder fills them.
"""

from __future__ import annotations

import json
from pathlib import Path

from minakanushi.architecture.config import SimulationConfig
from simulations.synthetic_world.dataset import Episode, generate_episode

SPLITS: tuple[str, ...] = ("train", "validation", "composition", "ood", "counterfactual")

SPLIT_SCENARIOS: dict[str, tuple[str, ...]] = {
    "train": ("const_velocity", "accelerate", "turn", "agent_move"),
    "validation": ("occlusion", "noisy", "obstacles"),
    "composition": ("accelerate", "turn", "occlusion", "agent_move"),
    "ood": ("gone_forever", "conflict", "hidden_correction"),
    "counterfactual": ("agent_move", "const_velocity"),
}


def episode_to_record(episode: Episode) -> dict:
    observations = []
    world_states = []
    actions = []
    for obs, truth in zip(episode.observations, episode.truth, strict=True):
        observations.append(
            {
                "timestamp": obs.timestamp,
                "agent_xy": list(obs.agent_xy),
                "agent_vel": list(obs.agent_vel),
                "visible_ids": [int(v["id"]) for v in obs.visible],
                "occluded_ids": list(obs.occluded_ids),
            }
        )
        world_states.append(
            {
                "event_time": truth.event_time,
                "entity_id": list(truth.entity_id),
                "kind": list(truth.kind),
                "xy": truth.xy.tolist(),
                "vel": truth.vel.tolist(),
            }
        )
        actions.append({"objective": truth.action, "target": list(truth.action_target)})
    return {
        "episode_id": f"{episode.scenario}-{episode.seed}-{episode.episode_index}",
        "seed": episode.seed,
        "scenario": episode.scenario,
        "self_state": {"platform": "synthetic", "kind": "agent"},
        "observations": observations,
        "world_states": world_states,
        "belief_states": [],
        "actions": actions,
        "future_branches": [],
        "events": [],
        "corrections": [],
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
        path.write_text(json.dumps(record), encoding="utf-8")
        paths.append(path)
    return paths
