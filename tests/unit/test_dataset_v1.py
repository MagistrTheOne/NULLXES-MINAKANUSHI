"""Dataset v1 contract: splits, replay identity, no millions."""

from __future__ import annotations

import json

from helpers import cpu_config
from simulations.synthetic_world.dataset import generate_episode
from simulations.synthetic_world.dataset_v1 import SPLITS, episode_to_record, write_split


def test_dataset_v1_contract_and_replay(tmp_path) -> None:
    sim = cpu_config().simulation
    ep = generate_episode(sim, seed=7, episode_index=0, length=6, scenario="accelerate")
    rec = episode_to_record(ep)
    for key in (
        "episode_id",
        "seed",
        "self_state",
        "observations",
        "world_states",
        "belief_states",
        "actions",
        "future_branches",
        "events",
        "corrections",
    ):
        assert key in rec
    again = generate_episode(sim, seed=7, episode_index=0, length=6, scenario="accelerate")
    assert episode_to_record(again)["world_states"][0]["xy"] == rec["world_states"][0]["xy"]
    paths = write_split(tmp_path, "ood", sim, seed=3, n_episodes=2, length=4)
    assert len(paths) == 2
    loaded = json.loads(paths[0].read_text(encoding="utf-8"))
    assert loaded["seed"] == 3
    assert set(SPLITS) == {"train", "validation", "composition", "ood", "counterfactual"}
