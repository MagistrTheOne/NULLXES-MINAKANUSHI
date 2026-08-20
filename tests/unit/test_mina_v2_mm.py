"""MINA V2 MM pack: organs → MinaUnit. Does not construct 6.8B."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from minakanushi.architecture.config import load_architecture
from minakanushi.architecture.freeze import assert_may_construct, is_6_8b_profile
from minakanushi.perception.bridge import Observation
from mina_v2 import (
    AudioEvent,
    EmbodimentState,
    MultimodalObservation,
    OperatorIntent,
    V2PerceptionBridge,
    VisionDetection,
    parse_operator,
    scene_branches,
)
from mina_v2.future import SceneBranch

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "models" / "MINA-V2-MM"


def _obs() -> Observation:
    return Observation(
        timestamp=1.0,
        agent_xy=(1.0, 1.0),
        agent_vel=(0.0, 0.0),
        heading=0.0,
        health=1.0,
        battery=0.9,
        visible=(),
    )


def test_organs_cpu_is_not_6_8b_profile() -> None:
    cfg = load_architecture(PACK / "configs" / "organs_cpu.yaml")
    assert not is_6_8b_profile(cfg)
    assert_may_construct(cfg, device="cpu")


def test_pack_architecture_yaml_is_not_6_8b() -> None:
    cfg = load_architecture(PACK / "architecture.yaml")
    assert not is_6_8b_profile(cfg)


def test_refuse_6_8b_construct_still_holds() -> None:
    cfg = load_architecture(ROOT / "configs" / "architecture" / "minakanushi_6_8b.yaml")
    with pytest.raises(RuntimeError, match="CPU"):
        assert_may_construct(cfg, device="cpu")


def test_four_organs_emit_expected_source_types() -> None:
    cfg = load_architecture(PACK / "configs" / "organs_cpu.yaml")
    bridge = V2PerceptionBridge(cfg)
    mm = MultimodalObservation(
        base=_obs(),
        detections=(
            VisionDetection(
                id=41,
                type="human",
                xy=(2.0, 3.0),
                vel=(0.1, 0.0),
                relation="approaching",
                event="moving",
                confidence=0.82,
            ),
        ),
        audio=(AudioEvent(kind="alarm", confidence=0.7),),
        operator="Найди машину возле склада",
        embodiment=EmbodimentState(battery=0.8, joints=(0.1, 0.2), force=0.0, balance=1.0),
    )
    units = bridge.encode(mm, device=torch.device("cpu"), dtype=torch.float32)
    by_source = {u.source_type: u for u in units if u.source_type in {"image", "text", "structured_event", "system_state"}}
    assert by_source["image"].kind == "mover"
    assert by_source["image"].metadata["type"] == "human"
    assert by_source["text"].metadata["target"] == "vehicle"
    assert by_source["text"].metadata["constraint"] == "warehouse"
    assert by_source["text"].metadata["action_intent"] is False
    assert by_source["structured_event"].metadata["channel"] == "audio"
    assert by_source["system_state"].metadata["pwm"] is False
    for unit in units:
        unit.validate()
        assert unit.semantic_embedding.numel() == cfg.latent_dim


def test_language_does_not_emit_action_intent() -> None:
    intent = parse_operator("Найди машину возле склада")
    assert intent == OperatorIntent(
        target="vehicle",
        constraint="warehouse",
        priority="search",
        utterance="Найди машину возле склада",
    )
    cfg = load_architecture(PACK / "configs" / "organs_cpu.yaml")
    units = V2PerceptionBridge(cfg).encode(
        MultimodalObservation(base=_obs(), operator=intent),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    text = [u for u in units if u.source_type == "text"]
    assert len(text) == 1
    assert "ActionIntent" not in type(text[0]).__name__
    assert text[0].metadata["action_intent"] is False


def test_scene_branch_is_not_video() -> None:
    branches = scene_branches(("human", "door", "robot", "box"), uncertainty=0.8)
    assert {b.branch_id for b in branches} == {"A", "B", "C"}
    hold = next(b for b in branches if b.branch_id == "C")
    assert hold.suggested_hold is True
    assert "SAFE_HOLD" in hold.narrative
    with pytest.raises(ValueError, match="video"):
        SceneBranch(
            branch_id="X",
            entities=("human",),
            relations=(),
            risk=0.1,
            uncertainty=0.1,
            suggested_hold=False,
            narrative="render video frames",
        )
