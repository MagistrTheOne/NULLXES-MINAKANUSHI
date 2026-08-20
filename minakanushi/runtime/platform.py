"""Plant adapter for the runtime loop. Not Gate 10 hardware embodiment.

Applies ActionIntent to SyntheticWorld. Does not emit PWM.
"""

from __future__ import annotations

from dataclasses import dataclass

from minakanushi.perception.bridge import Observation
from minakanushi.policy.intent import ActionIntent
from simulations.synthetic_world.world import Body, SyntheticWorld


@dataclass
class ActionResult:
    applied: bool
    objective: str
    agent_xy: tuple[float, float]
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "applied": self.applied,
            "objective": self.objective,
            "agent_xy": [float(self.agent_xy[0]), float(self.agent_xy[1])],
            "timestamp": float(self.timestamp),
        }


def _dump_body(body: Body) -> dict:
    return {
        "body_id": int(body.body_id),
        "kind": str(body.kind),
        "xy": body.xy.copy(),
        "vel": body.vel.copy(),
        "size": body.size.copy(),
        "accel": None if body.accel is None else body.accel.copy(),
    }


def _load_body(body: Body, raw: dict) -> None:
    body.body_id = int(raw["body_id"])
    body.kind = str(raw["kind"])
    body.xy = raw["xy"].copy()
    body.vel = raw["vel"].copy()
    body.size = raw["size"].copy()
    body.accel = None if raw.get("accel") is None else raw["accel"].copy()


class SyntheticPlatform:
    def __init__(self, world: SyntheticWorld) -> None:
        self.world = world

    def observe(self) -> Observation:
        return self.world.observe()

    def execute(self, intent: ActionIntent) -> ActionResult:
        self.world.step(intent)
        xy = self.world.agent.xy
        return ActionResult(
            applied=True,
            objective=str(intent.objective),
            agent_xy=(float(xy[0]), float(xy[1])),
            timestamp=float(self.world.t),
        )

    def dump(self) -> dict:
        return {
            "t": float(self.world.t),
            "last_intent": str(self.world.last_intent),
            "agent": _dump_body(self.world.agent),
            "movers": [_dump_body(b) for b in self.world.movers],
            "obstacles": [_dump_body(b) for b in self.world.obstacles],
            "targets": [_dump_body(b) for b in self.world.targets],
            "hidden_ids": sorted(int(i) for i in self.world.hidden_ids),
            "removed_ids": sorted(int(i) for i in self.world.removed_ids),
            "rng_state": self.world.rng.bit_generator.state,
        }

    def restore(self, raw: dict) -> None:
        self.world.t = float(raw["t"])
        self.world.last_intent = str(raw.get("last_intent", "SAFE_HOLD"))
        _load_body(self.world.agent, raw["agent"])
        for body, row in zip(self.world.movers, raw["movers"], strict=True):
            _load_body(body, row)
        for body, row in zip(self.world.obstacles, raw["obstacles"], strict=True):
            _load_body(body, row)
        for body, row in zip(self.world.targets, raw["targets"], strict=True):
            _load_body(body, row)
        self.world.hidden_ids = set(int(i) for i in raw.get("hidden_ids", ()))
        self.world.removed_ids = set(int(i) for i in raw.get("removed_ids", ()))
        if raw.get("rng_state") is not None:
            self.world.rng.bit_generator.state = raw["rng_state"]
