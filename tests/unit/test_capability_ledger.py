"""Capability protocol: measurements, not myths."""

from __future__ import annotations

from pathlib import Path

from minakanushi.training.capability import (
    ALLOWED_CLAIMS,
    FORBIDDEN_CLAIMS,
    LEDGER_ROWS,
    compare_heldout,
    compare_retention,
    cpu_trainer,
    forbidden_in_text,
    gate_a_retention,
    gate_b_heldout,
    gate_c_causality,
    gate_d_counterfactual,
    gate_e_memory,
    gate_f_revision_honesty,
    gate_g_no_shortcut,
    packet_snapshot,
    snapshot_drift,
)

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "docs" / "MINA_CAPABILITY_LEDGER.md"


def test_ledger_is_a_brake_not_a_myth() -> None:
    text = LEDGER.read_text(encoding="utf-8")
    assert "FORBIDDEN" in text
    assert "ALLOWED" in text
    for row in LEDGER_ROWS:
        assert row["ability"] in text
    for claim in ALLOWED_CLAIMS:
        assert claim in text
    for claim in (
        "MINA получила сознание",
        "MINA поняла мир",
        "MINA стала AGI",
        "модель думает как человек",
    ):
        assert claim in text
    assert "A lower loss is not a new ability" in text


def test_forbidden_scanner_catches_agi_story() -> None:
    assert forbidden_in_text("today MINA стала AGI") == ["MINA стала AGI"]
    assert forbidden_in_text("MINA улучшила prediction error") == []


def test_heldout_flags_memorization() -> None:
    before = {
        "seen_ADE": {"const_velocity": 4.0, "agent_move": 5.0},
        "unseen_ADE": {"const_velocity": 4.0, "agent_move": 5.0},
    }
    after = {
        "seen_ADE": {"const_velocity": 1.0, "agent_move": 1.0},
        "unseen_ADE": {"const_velocity": 4.2, "agent_move": 5.1},
    }
    verdict = compare_heldout(before, after)
    assert verdict["memorization"] is True
    assert verdict["pass"] is False


def test_retention_flags_forgotten_physics() -> None:
    before = {
        "scenarios": {
            "const_velocity": {"future_ADE": 1.0},
            "hidden_correction": {"future_ADE": 4.0},
        }
    }
    after = {
        "scenarios": {
            "const_velocity": {"future_ADE": 3.0},
            "hidden_correction": {"future_ADE": 1.5},
        }
    }
    verdict = compare_retention(before, after)
    assert verdict["physics_forgotten_for_hidden_trick"] is True
    assert verdict["pass"] is False


def test_capability_gates_on_cpu_dev(tmp_path: Path) -> None:
    trainer = cpu_trainer(12)
    a = gate_a_retention(trainer, tmp_path / "reference_before")
    assert a["pass"] is True
    assert (tmp_path / "reference_before" / "const_velocity.pt").is_file()
    b = gate_b_heldout(trainer)
    assert b["seen_seed"] == 7
    assert b["unseen_seed"] == 9999
    assert b["seen_ADE"]["const_velocity"] >= 0.0
    c = gate_c_causality(trainer)
    assert c["external_event"] == "unexpected_physics"
    assert c["pass"] is True
    assert "capability_proven" in c
    d = gate_d_counterfactual(trainer)
    assert d["pass"] is True
    assert d["future_distance"] > 0.0
    e = gate_e_memory(trainer, length=16)
    assert e["memory_ade_on"] >= 0.0
    assert e["memory_ade_off"] >= 0.0
    f = gate_f_revision_honesty(trainer)
    assert "never_revises_trap" in f
    g = gate_g_no_shortcut(trainer)
    assert g["pass"] is True
    assert "drop_vision" in g["future_ADE"]
    assert "permute_structure" in g["future_ADE"]
    assert "no_shortcut" in g
    pkt = trainer.unroll(1, scenario="const_velocity", episode_index=0, seed=7, length=12)
    same = packet_snapshot(pkt)
    drift = snapshot_drift(same, same)
    assert drift["world_state"] == 0.0
    assert drift["future_prediction"] == 0.0
