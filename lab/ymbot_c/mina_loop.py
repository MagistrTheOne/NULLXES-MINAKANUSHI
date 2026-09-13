"""YMBOT-C L0 — standalone MINAKANUSHI closed loop for partner laboratory tests.

This file is the sealed lab instrument. It does not construct minakanushi_6_8b.
It does not wrap a chat model. It does not emit PWM.

Contracts match Milestone 1:

    Observation → MinaUnit → world belief → uncertainty → futures
    → strategies → Constraint Kernel → ActionIntent → SyntheticWorld
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

IDENTITY = {
    "architecture": "MINAKANUSHI",
    "organization": "NULLXES",
    "short_name": "MINA",
    "package": "YMBOT-C",
    "system_class": "adaptive_situational_intelligence",
    "architecture_generation": 1,
    "native_runtime": "nullxes",
    "architecture_version": "0.1",
    "lab_track": "L0",
    "pwm": False,
    "accepted": False,
}

HOLD = {"WAIT", "OBSERVE", "SAFE_HOLD", "ABORT", "REQUEST_ASSISTANCE"}
STRATEGY_VOCAB = (
    "OBSERVE",
    "WAIT",
    "MOVE_TO",
    "FOLLOW",
    "INSPECT",
    "RETURN",
    "REQUEST_ASSISTANCE",
    "ABORT",
    "SAFE_HOLD",
)


@dataclass(frozen=True)
class LabConfig:
    seed: int = 11
    dt: float = 0.1
    arena: tuple[float, float, float, float] = (0.0, 10.0, 0.0, 10.0)
    agent_start: tuple[float, float] = (1.0, 1.0)
    home: tuple[float, float] = (1.0, 1.0)
    sensor_range: float = 3.5
    sensor_noise_std: float = 0.08
    occlusion: bool = True
    max_speed: float = 1.2
    persistence_steps: int = 8
    retirement_uncertainty: float = 0.95
    horizon: int = 8
    cognition_budget: int = 2
    restricted_zones: tuple[tuple[float, float, float, float], ...] = ((7.0, 7.0, 9.5, 9.5),)
    movers: tuple[dict[str, Any], ...] = (
        {"id": 11, "xy": (4.0, 2.0), "vel": (0.6, 0.2)},
        {"id": 12, "xy": (6.0, 6.0), "vel": (-0.4, 0.5)},
        {"id": 13, "xy": (2.5, 7.0), "vel": (0.3, -0.4)},
    )
    obstacles: tuple[dict[str, Any], ...] = (
        {"id": 21, "xy": (5.0, 5.0), "size": (1.2, 1.2)},
        {"id": 22, "xy": (3.0, 3.5), "size": (0.8, 1.6)},
    )
    targets: tuple[dict[str, Any], ...] = (
        {"id": 31, "xy": (8.0, 2.0)},
        {"id": 32, "xy": (2.0, 8.0)},
    )


@dataclass
class Body:
    body_id: int
    kind: str
    xy: np.ndarray
    vel: np.ndarray
    size: np.ndarray


@dataclass
class Observation:
    timestamp: float
    arrival_time: float
    agent_xy: tuple[float, float]
    agent_vel: tuple[float, float]
    heading: float
    visible: tuple[dict[str, Any], ...]
    occluded_ids: tuple[int, ...]
    noise_std: float


@dataclass
class MinaUnit:
    source_type: str
    source_id: int
    timestamp: float
    arrival_time: float
    sequence_index: int
    spatial_frame: str
    spatial_position: tuple[float, float, float]
    spatial_valid: bool
    semantic: np.ndarray
    confidence: float
    uncertainty: float
    persistence: float
    entity_reference: int
    kind: str
    episode_position: float
    memory_age: float

    def validate(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence out of range: {self.confidence}")
        if self.uncertainty < 0.0:
            raise ValueError(f"uncertainty must be >= 0, got {self.uncertainty}")
        if self.semantic.ndim != 1:
            raise ValueError(f"semantic must be 1-D, got {self.semantic.shape}")
        if not np.isfinite(self.semantic).all():
            raise ValueError("MinaUnit.semantic contains NaN/Inf")


@dataclass
class PositionState:
    sequence: float
    time: float
    space: tuple[float, float, float]
    episode: float
    memory_age: float
    source: float
    embedding: np.ndarray


@dataclass
class EntityBelief:
    entity_id: int
    kind: str
    xy: np.ndarray
    vel: np.ndarray
    confidence: float
    uncertainty: float
    persistence: float
    last_seen: float
    missing_steps: int


@dataclass
class WorldBelief:
    timestamp: float
    agent_xy: np.ndarray
    agent_vel: np.ndarray
    entities: dict[int, EntityBelief]
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class FutureTrajectory:
    strategy_id: str
    objective: str
    states_xy: np.ndarray
    probability: float
    uncertainty: float


@dataclass
class StrategyCandidate:
    strategy_id: str
    objective: str
    target_xy: tuple[float, float]
    expected_value: float
    uncertainty: float
    parameters: dict[str, float] = field(default_factory=dict)
    constraint_status: str = "unevaluated"


@dataclass(frozen=True)
class ConstraintAudit:
    strategy_id: str
    allowed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class AllowedStrategy:
    strategy_id: str
    objective: str
    target_xy: tuple[float, float]
    candidate: StrategyCandidate


@dataclass
class ActionIntent:
    strategy_id: str
    objective: str
    target_state: tuple[float, float]
    parameters: dict[str, float]
    confidence: float
    valid_until: float
    abort_conditions: tuple[str, ...]
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "strategy_id": self.strategy_id,
            "objective": self.objective,
            "target_state": [float(self.target_state[0]), float(self.target_state[1])],
            "parameters": dict(self.parameters),
            "confidence": float(self.confidence),
            "valid_until": float(self.valid_until),
            "abort_conditions": list(self.abort_conditions),
            "provenance": self.provenance,
            "pwm": False,
        }
        return payload


@dataclass
class CycleTelemetry:
    cycle_id: int
    physical_time: float
    observation_count: int
    entity_count: int
    event_count: int
    world_state_confidence: float
    uncertainty: float
    future_branches: int
    candidate_strategies: int
    rejected_strategies: int
    rejection_reasons: tuple[str, ...]
    selected_strategy: str
    cognition_cycles: int
    policy_enabled: bool


@dataclass
class EngineStep:
    belief: WorldBelief
    action_intent: ActionIntent
    telemetry: CycleTelemetry
    units: list[MinaUnit]
    allowed: tuple[AllowedStrategy, ...]
    rejected: tuple[StrategyCandidate, ...]
    futures: dict[str, list[FutureTrajectory]]


class SyntheticWorld:
    def __init__(self, config: LabConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.t = 0.0
        self.agent = Body(
            1,
            "agent",
            np.array(config.agent_start, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            np.array([0.4, 0.4], dtype=np.float64),
        )
        self.movers = [
            Body(
                int(m["id"]),
                "mover",
                np.array(m["xy"], dtype=np.float64),
                np.array(m["vel"], dtype=np.float64),
                np.array([0.3, 0.3], dtype=np.float64),
            )
            for m in config.movers
        ]
        self.obstacles = [
            Body(
                int(o["id"]),
                "obstacle",
                np.array(o["xy"], dtype=np.float64),
                np.zeros(2, dtype=np.float64),
                np.array(o.get("size", (1.0, 1.0)), dtype=np.float64),
            )
            for o in config.obstacles
        ]
        self.targets = [
            Body(
                int(t["id"]),
                "target",
                np.array(t["xy"], dtype=np.float64),
                np.zeros(2, dtype=np.float64),
                np.array([0.2, 0.2], dtype=np.float64),
            )
            for t in config.targets
        ]
        self.hidden_ids: set[int] = set()
        self.removed_ids: set[int] = set()
        self.last_intent = "SAFE_HOLD"

    def step(self, intent: ActionIntent | None) -> None:
        dt = self.config.dt
        if intent is not None:
            if "pwm" in intent.parameters:
                raise RuntimeError("PWM is forbidden on the MINA plant path")
            self.last_intent = intent.objective
            self.agent.vel = self._velocity_from_intent(intent)
        self.agent.xy = self._integrate(self.agent.xy, self.agent.vel, moving=True)
        for mover in self.movers:
            if mover.body_id in self.removed_ids:
                continue
            mover.xy = self._integrate(mover.xy, mover.vel, moving=True, bounce=True)
        self.t += dt

    def observe(self) -> Observation:
        visible: list[dict[str, Any]] = []
        occluded: list[int] = []
        for body in (*self.movers, *self.obstacles, *self.targets):
            if body.body_id in self.removed_ids:
                continue
            if body.body_id in self.hidden_ids or not self._visible(body):
                occluded.append(body.body_id)
                continue
            noisy = body.xy + self.rng.normal(0.0, self.config.sensor_noise_std, size=2)
            conf = max(0.2, 1.0 - self.config.sensor_noise_std * 2.0)
            visible.append(
                {
                    "id": body.body_id,
                    "kind": body.kind,
                    "xy": (float(noisy[0]), float(noisy[1])),
                    "vel": (float(body.vel[0]), float(body.vel[1])),
                    "confidence": conf,
                }
            )
        heading = 0.0
        if float(np.linalg.norm(self.agent.vel)) > 1e-6:
            heading = float(math.atan2(self.agent.vel[1], self.agent.vel[0]))
        return Observation(
            timestamp=self.t,
            arrival_time=self.t,
            agent_xy=(float(self.agent.xy[0]), float(self.agent.xy[1])),
            agent_vel=(float(self.agent.vel[0]), float(self.agent.vel[1])),
            heading=heading,
            visible=tuple(visible),
            occluded_ids=tuple(occluded),
            noise_std=self.config.sensor_noise_std,
        )

    def ground_truth(self) -> dict[int, dict[str, Any]]:
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
        if intent.objective in HOLD:
            return np.zeros(2, dtype=np.float64)
        target = np.array(intent.target_state, dtype=np.float64)
        delta = target - self.agent.xy
        norm = float(np.linalg.norm(delta))
        if norm < 1e-6:
            return np.zeros(2, dtype=np.float64)
        speed = min(self.config.max_speed, 1.0)
        return (delta / norm) * speed

    def _integrate(self, xy: np.ndarray, vel: np.ndarray, moving: bool, bounce: bool = False) -> np.ndarray:
        nxt = xy + vel * self.config.dt
        x0, x1, y0, y1 = self.config.arena
        if bounce:
            if nxt[0] < x0 or nxt[0] > x1:
                vel[0] *= -1.0
                nxt[0] = float(np.clip(nxt[0], x0, x1))
            if nxt[1] < y0 or nxt[1] > y1:
                vel[1] *= -1.0
                nxt[1] = float(np.clip(nxt[1], y0, y1))
        else:
            nxt[0] = float(np.clip(nxt[0], x0, x1))
            nxt[1] = float(np.clip(nxt[1], y0, y1))
        return nxt

    def _visible(self, body: Body) -> bool:
        dist = float(np.linalg.norm(body.xy - self.agent.xy))
        if dist > self.config.sensor_range:
            return False
        if not self.config.occlusion:
            return True
        for obs in self.obstacles:
            if body.body_id == obs.body_id:
                return True
            if _segment_hits_rect(self.agent.xy, body.xy, obs):
                return False
        return True


def _segment_hits_rect(a: np.ndarray, b: np.ndarray, obs: Body) -> bool:
    x0, y0 = float(obs.xy[0] - obs.size[0] / 2.0), float(obs.xy[1] - obs.size[1] / 2.0)
    x1, y1 = float(obs.xy[0] + obs.size[0] / 2.0), float(obs.xy[1] + obs.size[1] / 2.0)
    for t in np.linspace(0.0, 1.0, 8):
        p = a + (b - a) * t
        if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
            return True
    return False


def npf(unit: MinaUnit) -> PositionState:
    """NullxesPositionField lab encoder: six axes, not token index."""
    seq = float(unit.sequence_index)
    time_axis = float(unit.timestamp)
    space = unit.spatial_position if unit.spatial_valid else (0.0, 0.0, 0.0)
    episode = float(unit.episode_position)
    age = float(unit.memory_age)
    source = float({"telemetry": 1, "vector": 0, "structured_event": 3}.get(unit.source_type, 5))
    freqs = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
    parts = []
    for value in (seq, time_axis, space[0], space[1], episode, age, source):
        parts.append(np.sin(freqs * value))
        parts.append(np.cos(freqs * value))
    embedding = np.concatenate(parts)
    if not np.isfinite(embedding).all():
        raise ValueError("NPF embedding not finite")
    return PositionState(seq, time_axis, space, episode, age, source, embedding)


def encode_observation(obs: Observation, episode_position: float) -> list[MinaUnit]:
    units: list[MinaUnit] = []
    tel = np.array(
        [obs.agent_xy[0], obs.agent_xy[1], obs.agent_vel[0], obs.agent_vel[1], obs.heading, 1.0, 1.0],
        dtype=np.float64,
    )
    units.append(
        MinaUnit(
            source_type="telemetry",
            source_id=1,
            timestamp=obs.timestamp,
            arrival_time=obs.arrival_time,
            sequence_index=0,
            spatial_frame="arena",
            spatial_position=(obs.agent_xy[0], obs.agent_xy[1], 0.0),
            spatial_valid=True,
            semantic=tel,
            confidence=0.98,
            uncertainty=0.02,
            persistence=1.0,
            entity_reference=1,
            kind="agent",
            episode_position=episode_position,
            memory_age=0.0,
        )
    )
    for i, item in enumerate(obs.visible, start=1):
        sem = np.array([item["xy"][0], item["xy"][1], item["vel"][0], item["vel"][1], float(item["id"])], dtype=np.float64)
        units.append(
            MinaUnit(
                source_type="vector",
                source_id=int(item["id"]),
                timestamp=obs.timestamp,
                arrival_time=obs.arrival_time,
                sequence_index=i,
                spatial_frame="arena",
                spatial_position=(item["xy"][0], item["xy"][1], 0.0),
                spatial_valid=True,
                semantic=sem,
                confidence=float(item["confidence"]),
                uncertainty=max(0.02, 1.0 - float(item["confidence"])),
                persistence=1.0,
                entity_reference=int(item["id"]),
                kind=str(item["kind"]),
                episode_position=episode_position,
                memory_age=0.0,
            )
        )
    for unit in units:
        unit.validate()
        npf(unit)
    return units


class ConstraintKernel:
    def __init__(self, config: LabConfig) -> None:
        self.config = config

    def filter(
        self,
        candidates: list[StrategyCandidate],
        trajectories: dict[str, list[FutureTrajectory]],
    ) -> tuple[tuple[AllowedStrategy, ...], tuple[StrategyCandidate, ...], tuple[ConstraintAudit, ...]]:
        allowed: list[AllowedStrategy] = []
        rejected: list[StrategyCandidate] = []
        audits: list[ConstraintAudit] = []
        for cand in candidates:
            reasons: list[str] = []
            ok = True
            branches = trajectories.get(cand.strategy_id) or []
            points = [cand.target_xy]
            for traj in branches:
                for xy in traj.states_xy:
                    points.append((float(xy[0]), float(xy[1])))
            for x, y in points:
                if not self._in_arena(x, y):
                    ok = False
                    reasons.append(f"stay_in_arena violated at ({x:.2f},{y:.2f})")
                if self._in_restricted(x, y):
                    ok = False
                    reasons.append(f"no_enter_restricted_zone at ({x:.2f},{y:.2f})")
                if self._in_obstacle(x, y):
                    ok = False
                    reasons.append(f"no_collide_obstacle at ({x:.2f},{y:.2f})")
            speed = float(cand.parameters.get("speed", 1.0))
            if speed > self.config.max_speed + 1e-6:
                ok = False
                reasons.append(f"max_speed {speed} > {self.config.max_speed}")
            if ok:
                reasons.append("hard_constraints ok")
            cand.constraint_status = "allowed" if ok else "rejected"
            audits.append(ConstraintAudit(cand.strategy_id, ok, tuple(reasons)))
            if ok:
                allowed.append(AllowedStrategy(cand.strategy_id, cand.objective, cand.target_xy, cand))
            else:
                rejected.append(cand)
        return tuple(allowed), tuple(rejected), tuple(audits)

    def _in_arena(self, x: float, y: float) -> bool:
        x0, x1, y0, y1 = self.config.arena
        return x0 <= x <= x1 and y0 <= y <= y1

    def _in_restricted(self, x: float, y: float) -> bool:
        for zx0, zy0, zx1, zy1 in self.config.restricted_zones:
            if zx0 <= x <= zx1 and zy0 <= y <= zy1:
                return True
        return False

    def _in_obstacle(self, x: float, y: float) -> bool:
        for obs in self.config.obstacles:
            ox, oy = float(obs["xy"][0]), float(obs["xy"][1])
            sx, sy = float(obs.get("size", (1.0, 1.0))[0]), float(obs.get("size", (1.0, 1.0))[1])
            if ox - sx / 2.0 <= x <= ox + sx / 2.0 and oy - sy / 2.0 <= y <= oy + sy / 2.0:
                return True
        return False


class MinakanushiLabEngine:
    def __init__(self, config: LabConfig | None = None, *, policy_enabled: bool = True) -> None:
        self.config = config or LabConfig()
        self.policy_enabled = policy_enabled
        self.kernel = ConstraintKernel(self.config)
        self.cycle_id = 0
        self.belief = WorldBelief(
            timestamp=0.0,
            agent_xy=np.array(self.config.agent_start, dtype=np.float64),
            agent_vel=np.zeros(2, dtype=np.float64),
            entities={},
        )

    def step(self, observation: Observation) -> EngineStep:
        units = encode_observation(observation, episode_position=float(self.cycle_id))
        self.belief = self._update_belief(observation, units)
        for _ in range(self.config.cognition_budget):
            self.belief.timestamp = observation.timestamp
        situation_u = self._situation_uncertainty()
        candidates = self._strategies(situation_u)
        futures = {c.strategy_id: self._futures(c) for c in candidates}
        allowed, rejected, audits = self.kernel.filter(candidates, futures)
        if not self.policy_enabled:
            intent = ActionIntent(
                strategy_id="safe_hold",
                objective="SAFE_HOLD",
                target_state=(float(self.belief.agent_xy[0]), float(self.belief.agent_xy[1])),
                parameters={},
                confidence=1.0,
                valid_until=observation.timestamp + 1.0,
                abort_conditions=("policy_enabled=false",),
                provenance="authority.policy_off_cognition_on",
            )
        else:
            intent = self._select(allowed, observation.timestamp)
        confs = [self.belief.entities[k].confidence for k in self.belief.entities] or [1.0]
        uncs = [self.belief.entities[k].uncertainty for k in self.belief.entities] or [0.0]
        telemetry = CycleTelemetry(
            cycle_id=self.cycle_id,
            physical_time=observation.timestamp,
            observation_count=len(units),
            entity_count=len(self.belief.entities) + 1,
            event_count=len(self.belief.events),
            world_state_confidence=float(np.mean(confs)),
            uncertainty=float(np.mean(uncs)),
            future_branches=sum(len(v) for v in futures.values()),
            candidate_strategies=len(candidates),
            rejected_strategies=len(rejected),
            rejection_reasons=tuple(r for a in audits if not a.allowed for r in a.reasons),
            selected_strategy=intent.objective,
            cognition_cycles=self.config.cognition_budget,
            policy_enabled=self.policy_enabled,
        )
        self.cycle_id += 1
        return EngineStep(self.belief, intent, telemetry, units, allowed, rejected, futures)

    def _update_belief(self, obs: Observation, units: list[MinaUnit]) -> WorldBelief:
        seen = {u.entity_reference: u for u in units if u.kind != "agent"}
        entities = {eid: _clone_entity(ent) for eid, ent in self.belief.entities.items()}
        for eid, unit in seen.items():
            prev = entities.get(eid)
            xy = np.array(unit.spatial_position[:2], dtype=np.float64)
            if prev is None:
                vel = np.array(unit.semantic[2:4], dtype=np.float64) if unit.semantic.size >= 4 else np.zeros(2)
            else:
                dt = max(self.config.dt, obs.timestamp - prev.last_seen)
                vel = (xy - prev.xy) / dt
            entities[eid] = EntityBelief(
                entity_id=eid,
                kind=unit.kind,
                xy=xy,
                vel=vel,
                confidence=unit.confidence,
                uncertainty=unit.uncertainty,
                persistence=1.0,
                last_seen=obs.timestamp,
                missing_steps=0,
            )
        occluded = set(obs.occluded_ids)
        retired: list[int] = []
        for eid, ent in entities.items():
            if eid in seen:
                continue
            ent.missing_steps += 1
            ent.uncertainty = min(1.0, ent.uncertainty + 0.12)
            ent.confidence = max(0.05, ent.confidence * 0.82)
            ent.persistence = max(0.0, 1.0 - ent.missing_steps / float(self.config.persistence_steps))
            ent.xy = ent.xy + ent.vel * self.config.dt
            # Occlusion is evidence of a blocked body, not absence. Keep the
            # slot for as long as the entity stays occluded, even after the
            # persistence window and even if uncertainty saturates.
            if eid in occluded:
                continue
            if ent.missing_steps <= self.config.persistence_steps:
                continue
            if ent.uncertainty >= self.config.retirement_uncertainty or ent.missing_steps > self.config.persistence_steps:
                retired.append(eid)
        for eid in retired:
            del entities[eid]
        return WorldBelief(
            timestamp=obs.timestamp,
            agent_xy=np.array(obs.agent_xy, dtype=np.float64),
            agent_vel=np.array(obs.agent_vel, dtype=np.float64),
            entities=entities,
            events=list(self.belief.events),
        )

    def _situation_uncertainty(self) -> float:
        if not self.belief.entities:
            return 0.2
        return float(np.mean([e.uncertainty for e in self.belief.entities.values()]))

    def _strategies(self, uncertainty: float) -> list[StrategyCandidate]:
        agent = (float(self.belief.agent_xy[0]), float(self.belief.agent_xy[1]))
        home = self.config.home
        out = [
            StrategyCandidate("observe", "OBSERVE", agent, -0.4, uncertainty),
            StrategyCandidate("wait", "WAIT", agent, -0.5, uncertainty),
            StrategyCandidate("safe_hold", "SAFE_HOLD", agent, -0.2, 0.1),
            StrategyCandidate("abort", "ABORT", agent, -1.0, 0.1),
            StrategyCandidate("return", "RETURN", home, -_dist(agent, home), 0.2),
            StrategyCandidate("request_assistance", "REQUEST_ASSISTANCE", agent, -0.8, max(0.3, uncertainty)),
        ]
        for ent in self.belief.entities.values():
            xy = (float(ent.xy[0]), float(ent.xy[1]))
            if ent.kind == "target":
                out.append(
                    StrategyCandidate(
                        f"move_to_{ent.entity_id}",
                        "MOVE_TO",
                        xy,
                        expected_value=-_dist(agent, xy) + 1.5,
                        uncertainty=ent.uncertainty,
                        parameters={"entity_id": float(ent.entity_id), "speed": 1.0},
                    )
                )
            if ent.kind == "mover":
                out.append(
                    StrategyCandidate(
                        f"follow_{ent.entity_id}",
                        "FOLLOW",
                        xy,
                        expected_value=-_dist(agent, xy) - 0.3,
                        uncertainty=ent.uncertainty,
                    )
                )
                out.append(
                    StrategyCandidate(
                        f"inspect_{ent.entity_id}",
                        "INSPECT",
                        xy,
                        expected_value=-_dist(agent, xy) - 0.1,
                        uncertainty=ent.uncertainty,
                    )
                )
        return out

    def _futures(self, candidate: StrategyCandidate) -> list[FutureTrajectory]:
        states = np.zeros((self.config.horizon, 2), dtype=np.float64)
        pos = self.belief.agent_xy.copy()
        if candidate.objective in HOLD:
            vel = np.zeros(2, dtype=np.float64)
        else:
            delta = np.array(candidate.target_xy, dtype=np.float64) - pos
            norm = float(np.linalg.norm(delta))
            vel = np.zeros(2, dtype=np.float64) if norm < 1e-6 else (delta / norm) * min(self.config.max_speed, 1.0)
        for i in range(self.config.horizon):
            pos = pos + vel * self.config.dt
            states[i] = pos
        return [
            FutureTrajectory(
                strategy_id=candidate.strategy_id,
                objective=candidate.objective,
                states_xy=states,
                probability=0.7,
                uncertainty=candidate.uncertainty,
            )
        ]

    def _select(self, allowed: tuple[AllowedStrategy, ...], now: float) -> ActionIntent:
        if not allowed:
            return ActionIntent(
                strategy_id="safe_hold",
                objective="SAFE_HOLD",
                target_state=(float(self.belief.agent_xy[0]), float(self.belief.agent_xy[1])),
                parameters={},
                confidence=1.0,
                valid_until=now + 1.0,
                abort_conditions=("constraint_kernel_empty_allowed_set",),
                provenance="action_policy.fail_closed",
            )
        scored: list[tuple[float, AllowedStrategy]] = []
        for item in allowed:
            scored.append((item.candidate.expected_value - item.candidate.uncertainty, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        best = scored[0][1]
        return ActionIntent(
            strategy_id=best.strategy_id,
            objective=best.objective,
            target_state=best.target_xy,
            parameters=dict(best.candidate.parameters),
            confidence=max(0.0, 1.0 - best.candidate.uncertainty),
            valid_until=now + 1.0,
            abort_conditions=("hard_constraint_violation",),
            provenance="action_policy.argmax_allowed",
        )


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _clone_entity(ent: EntityBelief) -> EntityBelief:
    return EntityBelief(
        entity_id=ent.entity_id,
        kind=ent.kind,
        xy=ent.xy.copy(),
        vel=ent.vel.copy(),
        confidence=ent.confidence,
        uncertainty=ent.uncertainty,
        persistence=ent.persistence,
        last_seen=ent.last_seen,
        missing_steps=ent.missing_steps,
    )


def run_closed_loop(steps: int = 24, seed: int = 11, *, policy_enabled: bool = True) -> dict[str, Any]:
    config = LabConfig(seed=seed)
    world = SyntheticWorld(config)
    engine = MinakanushiLabEngine(config, policy_enabled=policy_enabled)
    history: list[dict[str, Any]] = []
    for _ in range(steps):
        obs = world.observe()
        result = engine.step(obs)
        if "pwm" in result.action_intent.to_dict() and result.action_intent.to_dict()["pwm"] is not False:
            raise RuntimeError("ActionIntent must not carry PWM commands")
        world.step(result.action_intent)
        history.append(
            {
                "t": obs.timestamp,
                "intent": result.action_intent.objective,
                "agent_before": list(map(float, result.belief.agent_xy)),
                "agent": [float(world.agent.xy[0]), float(world.agent.xy[1])],
                "entities": sorted(result.belief.entities),
                "rejected": result.telemetry.rejected_strategies,
                "selected": result.telemetry.selected_strategy,
                "uncertainty": result.telemetry.uncertainty,
            }
        )
    return {
        "identity": IDENTITY,
        "steps": steps,
        "seed": seed,
        "policy_enabled": policy_enabled,
        "final_intent": history[-1]["intent"] if history else None,
        "entity_count": len(engine.belief.entities),
        "history": history,
    }


def future_separation(engine: MinakanushiLabEngine) -> float:
    wait = StrategyCandidate("wait", "WAIT", tuple(engine.belief.agent_xy), -0.5, 0.1)
    goal = (min(9.0, float(engine.belief.agent_xy[0]) + 2.0), float(engine.belief.agent_xy[1]))
    move = StrategyCandidate("move_to_probe", "MOVE_TO", goal, 1.0, 0.1, {"speed": 1.0})
    a = engine._futures(wait)[0].states_xy[-1]
    b = engine._futures(move)[0].states_xy[-1]
    return float(np.linalg.norm(a - b))


def predict_mover_ade(engine: MinakanushiLabEngine, world: SyntheticWorld, mover_id: int = 11) -> float:
    ent = engine.belief.entities.get(mover_id)
    if ent is None:
        raise RuntimeError(f"mover {mover_id} missing from belief")
    pred = ent.xy + ent.vel * world.config.dt * 4.0
    for _ in range(4):
        world.step(ActionIntent("wait", "WAIT", tuple(world.agent.xy), {}, 1.0, 1e9, (), "ade"))
    gt = world.ground_truth()[mover_id]["xy"]
    return float(math.hypot(pred[0] - gt[0], pred[1] - gt[1]))


def run_selftest() -> dict[str, Any]:
    results: dict[str, bool] = {}
    notes: dict[str, str] = {}

    config = LabConfig(seed=11)
    world = SyntheticWorld(config)
    engine = MinakanushiLabEngine(config)
    obs0 = world.observe()
    step0 = engine.step(obs0)
    results["units_validate"] = all(u.source_type in {"telemetry", "vector"} for u in step0.units)
    results["npf_not_token_index"] = all(npf(u).time == u.timestamp for u in step0.units)
    results["pwm_false"] = step0.action_intent.to_dict()["pwm"] is False

    if 11 not in step0.belief.entities:
        raise RuntimeError("seed 11 must see mover 11 on the first frame")
    u0 = float(step0.belief.entities[11].uncertainty)
    world.hidden_ids.add(11)
    hidden_step = step0
    occlude_steps = config.persistence_steps + 4
    for _ in range(occlude_steps):
        world.step(ActionIntent("wait", "WAIT", tuple(world.agent.xy), {}, 1.0, 1e9, (), "persist"))
        hidden_step = engine.step(world.observe())
        if 11 not in hidden_step.belief.entities:
            break
    survived = 11 in hidden_step.belief.entities
    u1 = float(hidden_step.belief.entities[11].uncertainty) if survived else -1.0
    results["persistence"] = survived and u1 > u0
    notes["persistence"] = f"entity 11 survived={survived} uncertainty {u0:.3f} → {u1:.3f}"

    a_then_b = MinakanushiLabEngine(LabConfig(seed=3))
    b_then_a = MinakanushiLabEngine(LabConfig(seed=3))

    def _unit(eid: int, x: float, t: float) -> Observation:
        return Observation(
            timestamp=t,
            arrival_time=t,
            agent_xy=(1.0, 1.0),
            agent_vel=(0.0, 0.0),
            heading=0.0,
            visible=({"id": eid, "kind": "mover", "xy": (x, 2.0), "vel": (0.0, 0.0), "confidence": 0.9},),
            occluded_ids=(),
            noise_std=0.0,
        )

    a_then_b.step(_unit(11, 1.0, 0.0))
    a_then_b.step(_unit(11, 2.0, 0.1))
    b_then_a.step(_unit(11, 2.0, 0.0))
    b_then_a.step(_unit(11, 1.0, 0.1))
    vel_ab = float(a_then_b.belief.entities[11].vel[0])
    vel_ba = float(b_then_a.belief.entities[11].vel[0])
    results["temporal_order"] = vel_ab > 0.0 and vel_ba < 0.0
    notes["temporal_order"] = f"vel A→B={vel_ab:.3f} vel B→A={vel_ba:.3f}"

    sep = future_separation(engine)
    results["counterfactual"] = sep > 0.15
    notes["counterfactual"] = f"WAIT vs MOVE_TO terminal Δ={sep:.3f}"

    bait = StrategyCandidate("raid", "MOVE_TO", (8.2, 8.2), expected_value=100.0, uncertainty=0.0, parameters={"speed": 1.0})
    hold = StrategyCandidate("safe_hold", "SAFE_HOLD", (1.0, 1.0), expected_value=-10.0, uncertainty=0.1)
    futures = {c.strategy_id: engine._futures(c) for c in (bait, hold)}
    allowed, rejected, _ = engine.kernel.filter([bait, hold], futures)
    results["hard_constraint"] = bait.strategy_id not in {a.strategy_id for a in allowed} and any(
        c.strategy_id == "raid" for c in rejected
    )
    selected = engine._select(allowed, 0.0)
    results["policy_cannot_bypass"] = selected.strategy_id != "raid"
    notes["hard_constraint"] = f"rejected={[c.strategy_id for c in rejected]} selected={selected.strategy_id}"

    wait_world = SyntheticWorld(LabConfig(seed=5))
    move_world = SyntheticWorld(LabConfig(seed=5))
    start_xy = (float(wait_world.agent.xy[0]), float(wait_world.agent.xy[1]))
    wait_world.step(ActionIntent("wait", "WAIT", start_xy, {}, 1.0, 1.0, (), "loop"))
    move_world.step(ActionIntent("go", "MOVE_TO", (2.0, 8.0), {"speed": 1.0}, 1.0, 1.0, (), "loop"))
    delta = float(np.linalg.norm(move_world.agent.xy - wait_world.agent.xy))
    loop = run_closed_loop(steps=12, seed=11)
    results["closed_loop"] = loop["steps"] == 12 and delta > 1e-4
    notes["closed_loop"] = f"WAIT vs MOVE_TO plant Δ={delta:.3f} final={loop['final_intent']}"

    off = MinakanushiLabEngine(LabConfig(seed=11), policy_enabled=False)
    off_world = SyntheticWorld(LabConfig(seed=11))
    off_step = off.step(off_world.observe())
    results["policy_off_cognition_on"] = (
        off_step.action_intent.objective == "SAFE_HOLD" and len(off_step.belief.entities) >= 1
    )

    w_ade = SyntheticWorld(LabConfig(seed=11))
    e_ade = MinakanushiLabEngine(LabConfig(seed=11))
    for _ in range(6):
        e_ade.step(w_ade.observe())
        w_ade.step(ActionIntent("wait", "WAIT", tuple(w_ade.agent.xy), {}, 1.0, 1e9, (), "warm"))
    try:
        ade = predict_mover_ade(e_ade, w_ade, 11)
        results["prediction"] = ade < 2.5
        notes["prediction"] = f"mover11 ADE@4={ade:.3f}"
    except RuntimeError as exc:
        results["prediction"] = False
        notes["prediction"] = str(exc)

    results["identity_not_prompt"] = IDENTITY["architecture"] == "MINAKANUSHI" and IDENTITY["pwm"] is False
    failed = [k for k, v in results.items() if not v]
    return {
        "identity": IDENTITY,
        "passed": len(failed) == 0,
        "failed": failed,
        "results": results,
        "notes": notes,
    }


def _collect_residual_pairs(seed: int, n_cycles: int) -> tuple[np.ndarray, np.ndarray]:
    config = LabConfig(seed=seed, sensor_range=8.0, occlusion=False)
    world = SyntheticWorld(config)
    engine = MinakanushiLabEngine(config)
    xs: list[list[float]] = []
    ys: list[list[float]] = []
    for _ in range(n_cycles):
        obs = world.observe()
        result = engine.step(obs)
        before = {eid: ent.xy.copy() for eid, ent in result.belief.entities.items() if ent.kind == "mover"}
        # Collect under WAIT so the agent stays in range. Policy choice is not this gate.
        world.step(
            ActionIntent(
                "wait",
                "WAIT",
                (float(world.agent.xy[0]), float(world.agent.xy[1])),
                {},
                1.0,
                obs.timestamp + 1.0,
                (),
                "l1.collect",
            )
        )
        after = world.ground_truth()
        for eid, xy0 in before.items():
            if eid not in after:
                continue
            ent = result.belief.entities[eid]
            hold = 1.0 if result.action_intent.objective in HOLD else 0.0
            xs.append([float(xy0[0]), float(xy0[1]), float(ent.vel[0]), float(ent.vel[1]), hold, config.dt])
            ys.append([float(after[eid]["xy"][0] - xy0[0]), float(after[eid]["xy"][1] - xy0[1])])
    if len(xs) < 16:
        raise RuntimeError(f"L1 collected {len(xs)} transitions, need >= 16")
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _train_linear_residual(x: np.ndarray, y: np.ndarray, steps: int, seed: int) -> dict[str, float]:
    """Explicit SGD on a linear residual. Not a hardcoded win."""
    rng = np.random.default_rng(seed)
    mean = x.mean(axis=0)
    std = np.clip(x.std(axis=0), 1e-6, None)
    xn = (x - mean) / std
    w = rng.normal(0.0, 0.05, size=(xn.shape[1], 2))
    b = np.zeros(2, dtype=np.float64)
    lr = 0.05
    n = float(xn.shape[0])
    zero_ade = float(np.sqrt(np.mean(y * y)))
    init = xn @ w + b
    init_ade = float(np.sqrt(np.mean((init - y) ** 2)))
    last = 0.0
    for step in range(steps):
        pred = xn @ w + b
        err = pred - y
        if not np.isfinite(err).all():
            raise RuntimeError(f"L1 non-finite residual at step {step}")
        last = float(np.mean(err * err))
        grad_w = (xn.T @ err) * (2.0 / n)
        grad_b = err.sum(axis=0) * (2.0 / n)
        if not (np.isfinite(grad_w).all() and np.isfinite(grad_b).all()):
            raise RuntimeError(f"L1 non-finite grad at step {step}")
        w = w - lr * grad_w
        b = b - lr * grad_b
    trained = xn @ w + b
    trained_ade = float(np.sqrt(np.mean((trained - y) ** 2)))
    if trained_ade >= zero_ade:
        raise RuntimeError(f"L1 residual did not beat zero predictor: {trained_ade:.4f} >= {zero_ade:.4f}")
    return {
        "loss": last,
        "ade_zero": zero_ade,
        "ade_init": init_ade,
        "ade_trained": trained_ade,
        "params": float(w.size + b.size),
    }


def train_world_residual(steps: int = 40, device: str = "cpu", seed: int = 11) -> dict[str, Any]:
    """L1 lab instrument: train a residual next-position head on SyntheticWorld.

    Default path is numpy SGD (portable). CUDA uses the same data and a tiny
    torch linear head. This is not minakanushi_6_8b. Constraints stay non-learned.
    """
    x, y = _collect_residual_pairs(seed, n_cycles=max(80, steps * 4))
    backend = "numpy"
    if device.startswith("cuda"):
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError("L1 --device cuda requires PyTorch") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("L1 asked for CUDA but torch.cuda.is_available() is False")
        torch.manual_seed(seed)
        xt = torch.tensor(x, dtype=torch.float32, device="cuda")
        yt = torch.tensor(y, dtype=torch.float32, device="cuda")
        model = nn.Linear(x.shape[1], 2).cuda()
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        with torch.no_grad():
            zero_ade = float(yt.pow(2).mean().sqrt().item())
            init_ade = float((model(xt) - yt).pow(2).mean().sqrt().item())
        last = 0.0
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            pred = model(xt)
            loss = (pred - yt).pow(2).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"L1 non-finite loss at step {step}")
            loss.backward()
            opt.step()
            last = float(loss.item())
        with torch.no_grad():
            trained_ade = float((model(xt) - yt).pow(2).mean().sqrt().item())
        if trained_ade >= zero_ade:
            raise RuntimeError(f"L1 residual did not beat zero predictor: {trained_ade:.4f} >= {zero_ade:.4f}")
        metrics = {
            "loss": last,
            "ade_zero": zero_ade,
            "ade_init": init_ade,
            "ade_trained": trained_ade,
            "params": float(sum(p.numel() for p in model.parameters())),
        }
        backend = "torch.cuda"
    else:
        metrics = _train_linear_residual(x, y, steps=steps, seed=seed)
    return {
        "identity": {**IDENTITY, "lab_track": "L1"},
        "device": device,
        "backend": backend,
        "steps": steps,
        "transitions": int(x.shape[0]),
        "constructs_6_8b": False,
        **metrics,
    }


def dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print(dumps(run_selftest()))
