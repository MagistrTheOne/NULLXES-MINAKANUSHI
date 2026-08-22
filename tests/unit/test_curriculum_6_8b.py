"""6.8B episode curriculum: observation → action → transition → lesson."""

from __future__ import annotations

import json
from pathlib import Path

from minakanushi.architecture.config import load_simulation
from simulations.synthetic_world.curriculum_6_8b import (
    PHASE_ORDER,
    episode_record,
    write_curriculum,
)
from simulations.synthetic_world.replay import records_identical

ROOT = Path(__file__).resolve().parents[2]


def test_each_phase_emits_transitions_not_tokens(tmp_path: Path) -> None:
    config = load_simulation(ROOT / "configs" / "simulation" / "milestone1.yaml")
    for phase in PHASE_ORDER:
        record = episode_record(config, phase=phase, seed=7, episode_index=0, length=8)
        assert record["phase"] == phase
        assert record["transitions"]
        step = record["transitions"][0]
        for key in ("observation_t", "belief_t", "action_t", "observation_t1", "lesson"):
            assert key in step
        assert record["embodiment"]["pwm"] is False
        assert "token" not in record


def test_curriculum_replay_identity(tmp_path: Path) -> None:
    config = load_simulation(ROOT / "configs" / "simulation" / "milestone1.yaml")
    a = episode_record(config, phase="causality", seed=7, episode_index=1, length=8)
    b = episode_record(config, phase="causality", seed=7, episode_index=1, length=8)
    assert records_identical(a, b)


def test_write_curriculum_creates_four_phases(tmp_path: Path) -> None:
    config = load_simulation(ROOT / "configs" / "simulation" / "milestone1.yaml")
    written = write_curriculum(tmp_path, config, seed=7, n_episodes=1, length=8)
    assert tuple(written) == PHASE_ORDER
    for phase, paths in written.items():
        assert len(paths) == 1
        assert paths[0].is_file()
    assert (tmp_path / "index.jsonl").is_file()
    first = json.loads((tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "scenario" in first


def test_v02_phases_include_causal_agency_embodiment() -> None:
    from simulations.synthetic_world.curriculum_6_8b import PHASES
    from simulations.synthetic_world.dataset import generate_episode

    assert "brake" in PHASES["physics"]
    assert "goal_change" in PHASES["agency"]
    assert "follow" in PHASES["agency"]
    assert "avoid" in PHASES["agency"]
    assert "hidden_correction" in PHASES["causality"]
    assert "hidden_object" in PHASES["causality"]
    assert "motor_delay" in PHASES["embodiment"]
    assert "sensor_delay" in PHASES["embodiment"]
    config = load_simulation(ROOT / "configs" / "simulation" / "milestone1.yaml")
    for scenario in ("brake", "goal_change", "unexpected_stop", "motor_delay", "follow", "avoid", "wrong_velocity"):
        episode = generate_episode(config, seed=7, episode_index=0, length=8, scenario=scenario)
        assert episode.scenario == scenario
        assert len(episode.observations) == 8
        assert episode.truth[0].action in {"WAIT", "MOVE_TO", "FOLLOW", "AVOID"}


def test_audit_curriculum_report(tmp_path: Path) -> None:
    import importlib.util

    config = load_simulation(ROOT / "configs" / "simulation" / "milestone1.yaml")
    write_curriculum(tmp_path, config, seed=7, n_episodes=1, length=8)
    path = ROOT / "scripts" / "audit_curriculum.py"
    spec = importlib.util.spec_from_file_location("audit_curriculum", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    report = mod.audit_curriculum(tmp_path)
    assert report["n_episodes"] == 4
    assert report["pwm"] is False
    assert report["hf_role"] == "adapter_only"
    assert set(report["phase_counts"]) == {"physics", "agency", "causality", "embodiment"}
    assert report["gate"]["pwm_false"] is True
    assert "future_diversity_min" in report
    assert "future_diversity_max" in report
    assert "future_diversity_std" in report
    assert "revision_distribution" in report
    assert "max_action_fraction" in report
    assert "decision_entropy" in report
    assert "wait_required" in report
    assert "wait_safe_button" in report
    assert (tmp_path / "train" / "index.jsonl").is_file()
    assert (tmp_path / "heldout" / "index.jsonl").is_file()
    assert "future_diversity_collapsed" in report


def test_v03_default_lengths_and_counterfactual_forks() -> None:
    from simulations.synthetic_world.curriculum_6_8b import PHASE_LENGTHS, episode_record
    from simulations.synthetic_world.dataset import revision_frame

    assert PHASE_LENGTHS == {"physics": 32, "agency": 32, "causality": 64, "embodiment": 64}
    assert revision_frame("hidden_correction", 64) > 6
    config = load_simulation(ROOT / "configs" / "simulation" / "milestone1.yaml")
    rec = episode_record(config, phase="causality", seed=7, episode_index=0)
    assert rec["curriculum"] == "mina_6_8b_v03"
    assert rec["future_diversity"] > 1e-6
    assert set(rec["counterfactuals"]["strategies"]) == {"WAIT", "MOVE_TO", "FOLLOW", "AVOID"}
    assert rec["embodiment"]["pwm"] is False
    assert any(row.get("correction_type") for row in rec["corrections"])
    follow = episode_record(config, phase="agency", seed=7, episode_index=4, length=8)
    assert follow["scenario"] == "follow"
    assert follow["actions"][0]["objective"] == "FOLLOW"
    long_rec = episode_record(config, phase="causality", seed=7, episode_index=0, length=64)
    short_rec = episode_record(config, phase="causality", seed=7, episode_index=0, length=8)
    assert len(long_rec["corrections"]) > len(short_rec["corrections"])
    kinds = {str(row.get("correction_type")) for row in long_rec["corrections"] if row.get("correction_type")}
    assert kinds & {"hidden_object", "wrong_intent"}

