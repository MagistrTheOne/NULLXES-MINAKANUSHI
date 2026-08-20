"""SyntheticWorld — deterministic physical environment for Milestone 1.

Contains one MINAKANUSHI-controlled agent, moving entities, static obstacles,
targets, partial observations, sensor noise, and hard no-go zones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from minakanushi.architecture.config import SimulationConfig
from minakanushi.perception.bridge import Observation
from minakanushi.policy.intent import ActionIntent
from minakanushi.strategy.hold import is_hold


@dataclass
class Body:
    body_id: int
    kind: str
    xy: np.ndarray
    vel: np.ndarray
    size: np.ndarray
    accel: np.ndarray | None = None


class SyntheticWorld:
    def __init__(self, config: SimulationConfig, seed: int = 7) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.agent = Body(1, "agent", np.array(config.agent_start, dtype=np.float64), np.zeros(2), np.array([0.4, 0.4]))
        self.movers = [
            Body(int(m["id"]), "mover", np.array(m["xy"], dtype=np.float64), np.array(m["vel"], dtype=np.float64), np.array([0.3, 0.3]))
            for m in config.movers
        ]
        self.obstacles = [
            Body(int(o["id"]), "obstacle", np.array(o["xy"], dtype=np.float64), np.zeros(2), np.array(o.get("size", [1.0, 1.0]), dtype=np.float64))
            for o in config.obstacles
        ]
        self.targets = [
            Body(int(t["id"]), "target", np.array(t["xy"], dtype=np.float64), np.zeros(2), np.array([0.2, 0.2]))
            for t in config.targets
        ]
        self.last_intent = "SAFE_HOLD"
        self.hidden_ids: set[int] = set()
        self.removed_ids: set[int] = set()

    def step(self, intent: ActionIntent | None) -> None:
        dt = self.config.dt
        if intent is not None:
            self.last_intent = intent.objective
            self.agent.vel = self._velocity_from_intent(intent)
        self.agent.xy = self._integrate(self.agent.xy, self.agent.vel, dt, moving=True)
        for mover in self.movers:
            if mover.body_id in self.removed_ids:
                continue
            if mover.accel is not None:
                mover.vel = mover.vel + mover.accel * dt
            mover.xy = self._integrate(mover.xy, mover.vel, dt, moving=True, bounce=True)
        self.t += dt

    def observe(self) -> Observation:
        visible = []
        occluded = []
        for body in [*self.movers, *self.obstacles, *self.targets]:
            if body.body_id in self.removed_ids:
                continue
            if body.body_id in self.hidden_ids:
                occluded.append(body.body_id)
                continue
            if self._visible(body):
                noisy_xy = body.xy + self.rng.normal(0.0, self.config.sensor_noise_std, size=2)
                conf = max(0.2, 1.0 - self.config.sensor_noise_std * 2.0)
                visible.append(
                    {
                        "id": body.body_id,
                        "kind": body.kind,
                        "xy": (float(noisy_xy[0]), float(noisy_xy[1])),
                        "vel": (float(body.vel[0]), float(body.vel[1])),
                        "confidence": conf,
                    }
                )
            else:
                occluded.append(body.body_id)
        return Observation(
            timestamp=self.t,
            agent_xy=(float(self.agent.xy[0]), float(self.agent.xy[1])),
            agent_vel=(float(self.agent.vel[0]), float(self.agent.vel[1])),
            heading=float(math.atan2(self.agent.vel[1], self.agent.vel[0])) if np.linalg.norm(self.agent.vel) > 1e-6 else 0.0,
            health=1.0,
            battery=1.0,
            visible=tuple(visible),
            occluded_ids=tuple(occluded),
            noise_std=self.config.sensor_noise_std,
        )

    def ground_truth(self) -> dict[int, dict[str, object]]:
        bodies = [self.agent, *self.movers, *self.obstacles, *self.targets]
        return {
            b.body_id: {
                "kind": b.kind,
                "xy": (float(b.xy[0]), float(b.xy[1])),
                "vel": (float(b.vel[0]), float(b.vel[1])),
            }
            for b in bodies
            if b.body_id not in self.removed_ids
        }

    def _velocity_from_intent(self, intent: ActionIntent) -> np.ndarray:
        if is_hold(intent.objective):
            return np.zeros(2)
        target = np.array(intent.target_state, dtype=np.float64)
        delta = target - self.agent.xy
        norm = np.linalg.norm(delta)
        if norm < 1e-6:
            return np.zeros(2)
        speed = min(self.config.max_speed, 1.0)
        return (delta / norm) * speed

    def _integrate(self, xy: np.ndarray, vel: np.ndarray, dt: float, moving: bool, bounce: bool = False) -> np.ndarray:
        nxt = xy + vel * dt
        x0, x1, y0, y1 = self.config.arena
        if bounce:
            if nxt[0] < x0 or nxt[0] > x1:
                vel[0] *= -1
                nxt[0] = float(np.clip(nxt[0], x0, x1))
            if nxt[1] < y0 or nxt[1] > y1:
                vel[1] *= -1
                nxt[1] = float(np.clip(nxt[1], y0, y1))
        else:
            nxt[0] = float(np.clip(nxt[0], x0, x1))
            nxt[1] = float(np.clip(nxt[1], y0, y1))
        return nxt

    def _visible(self, body: Body) -> bool:
        delta = body.xy - self.agent.xy
        dist = float(np.linalg.norm(delta))
        if dist > self.config.sensor_range:
            return False
        if not self.config.occlusion:
            return True
        for obs in self.obstacles:
            if body.body_id == obs.body_id:
                return True
            if self._segment_hits_rect(self.agent.xy, body.xy, obs):
                return False
        return True

    def _segment_hits_rect(self, a: np.ndarray, b: np.ndarray, obs: Body) -> bool:
        x0 = obs.xy[0] - obs.size[0] / 2
        x1 = obs.xy[0] + obs.size[0] / 2
        y0 = obs.xy[1] - obs.size[1] / 2
        y1 = obs.xy[1] + obs.size[1] / 2
        samples = np.linspace(0.0, 1.0, 8)
        for s in samples[1:-1]:
            p = a * (1 - s) + b * s
            if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
                return True
        return False
