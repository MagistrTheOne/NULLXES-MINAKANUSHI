"""Checkpoint selection must use numeric step, not lexicographic names."""

from __future__ import annotations

from pathlib import Path

from minakanushi.training.checkpoint import checkpoint_step, latest_mina

ROOT = Path(__file__).resolve().parents[2]


def test_lexicographic_sort_picks_the_wrong_step() -> None:
    names = [
        "minakanushi_stage0_step9.mina",
        "minakanushi_stage0_step100.mina",
        "minakanushi_stage0_step20.mina",
    ]
    assert sorted(names)[-1] == "minakanushi_stage0_step9.mina"
    assert max(names, key=checkpoint_step) == "minakanushi_stage0_step100.mina"


def test_latest_mina_selects_highest_step_number(tmp_path) -> None:
    (tmp_path / "minakanushi_stage0_step9.mina").write_bytes(b"mina")
    (tmp_path / "minakanushi_stage0_step100.mina").write_bytes(b"mina")
    (tmp_path / "minakanushi_stage0_step20.mina").write_bytes(b"mina")
    chosen = latest_mina(tmp_path)
    assert chosen.name == "minakanushi_stage0_step100.mina"


def test_gate02_post_delegates_to_numeric_latest_mina() -> None:
    text = (ROOT / "scripts" / "gate02_post.py").read_text(encoding="utf-8")
    assert "return latest_mina(OUT)" in text
    assert 'sorted(OUT.glob("*.mina"))' not in text
