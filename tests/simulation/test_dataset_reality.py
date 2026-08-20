"""Gate 08.5 Dataset Reality: replay, inspector, balance, causal sanity. Not Gate 09."""

from __future__ import annotations

import numpy as np

from helpers import cpu_config
from minakanushi.policy.intent import ActionIntent
from simulations.synthetic_world.balance import load_records, tally_records
from simulations.synthetic_world.dataset import generate_episode
from simulations.synthetic_world.dataset_v1 import SPLIT_SCENARIOS, episode_to_record, write_split
from simulations.synthetic_world.inspector import format_episode
from simulations.synthetic_world.replay import canonical_json, records_identical
from simulations.synthetic_world.world import SyntheticWorld


def _intent(name: str, xy: tuple[float, float]) -> ActionIntent:
    return ActionIntent(name.lower(), name, xy, {}, 1.0, 1e9, (), "dataset_reality")


def test_08_5a_replay_is_identical() -> None:
    sim = cpu_config().simulation
    keys = (
        "world_states",
        "observations",
        "actions",
        "events",
        "future_branches",
        "belief_states",
        "outcomes",
        "corrections",
    )
    for scenario in ("accelerate", "conflict", "occlusion", "agent_move"):
        a = episode_to_record(generate_episode(sim, seed=11, episode_index=2, length=8, scenario=scenario))
        b = episode_to_record(generate_episode(sim, seed=11, episode_index=2, length=8, scenario=scenario))
        assert records_identical(a, b)
        assert canonical_json(a) == canonical_json(b)
        for key in keys:
            assert a[key] == b[key]
            assert key in a


def test_08_5b_inspector_shows_time_action_reality() -> None:
    sim = cpu_config().simulation
    rec = episode_to_record(generate_episode(sim, seed=4, episode_index=0, length=6, scenario="hidden_correction"))
    text = format_episode(rec, max_frames=4)
    assert "TIME 0" in text
    assert "Entities:" in text
    assert "Belief:" in text
    assert "Action:" in text
    assert "Predicted:" in text
    assert "REALITY:" in text
    assert "Correction:" in text
    assert rec["scenario"] in text


def test_08_5c_balance_not_constant_velocity_only(tmp_path) -> None:
    sim = cpu_config().simulation
    write_split(tmp_path, "composition", sim, seed=7, n_episodes=12, length=6)
    write_split(tmp_path, "ood", sim, seed=7, n_episodes=9, length=6)
    records = load_records(tmp_path)
    report = tally_records(records)
    assert report.n_episodes == 21
    assert report.max_scenario_fraction() <= 0.34
    assert report.occlusion_count > 0
    assert report.action_count
    assert "WAIT" in report.action_count or "MOVE_TO" in report.action_count
    ood_scenarios = {r["scenario"] for r in records if r["scenario"] in SPLIT_SCENARIOS["ood"]}
    assert "conflict" in ood_scenarios
    assert report.conflict_count > 0
    assert not report.constant_velocity_collapsed()
    assert len(report.scenario_count) >= 6


def test_08_5d_causal_sanity_wait_vs_move() -> None:
    sim = cpu_config().simulation
    wait_world = SyntheticWorld(sim, seed=21)
    move_world = SyntheticWorld(sim, seed=21)
    far = np.array([9.0, 9.0], dtype=np.float64)
    wait_world.movers[0].xy = far.copy()
    move_world.movers[0].xy = far.copy()
    wait_world.movers[0].vel = np.zeros(2)
    move_world.movers[0].vel = np.zeros(2)
    home = (float(wait_world.agent.xy[0]), float(wait_world.agent.xy[1]))
    target = (8.0, 2.0)
    for _ in range(6):
        wait_world.step(_intent("WAIT", home))
        move_world.step(_intent("MOVE_TO", target))
    agent_div = float(np.linalg.norm(wait_world.agent.xy - move_world.agent.xy))
    mover_div = float(np.linalg.norm(wait_world.movers[0].xy - move_world.movers[0].xy))
    assert agent_div > 0.4
    assert mover_div < 1e-9
