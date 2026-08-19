"""Constraint rule definitions. Hard constraints cannot be overridden."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from minakanushi.architecture.config import SimulationConfig
from minakanushi.future.trajectory import FutureTrajectory
from minakanushi.state.entity import AGENT_SLOT
from minakanushi.strategy.candidate import StrategyCandidate


class ConstraintClass(str, Enum):
    HARD = "HARD"
    SOFT = "SOFT"
    MISSION = "MISSION"
    RESOURCE = "RESOURCE"
    ENVIRONMENTAL = "ENVIRONMENTAL"
    OPERATIONAL = "OPERATIONAL"


@dataclass(frozen=True)
class ConstraintRule:
    name: str
    cls: ConstraintClass
    description: str

    def evaluate(
        self,
        candidate: StrategyCandidate,
        trajectory: FutureTrajectory | None,
        simulation: SimulationConfig,
    ) -> tuple[bool, str]:
        raise RuntimeError(f"rule {self.name} missing evaluate()")


class StayInArena(ConstraintRule):
    def __init__(self) -> None:
        super().__init__("stay_in_arena", ConstraintClass.HARD, "agent must remain inside arena")

    def evaluate(self, candidate, trajectory, simulation):
        x0, x1, y0, y1 = simulation.arena
        points = [candidate.target_xy]
        if trajectory is not None:
            agent = trajectory.states_xy[:, AGENT_SLOT]
            points.extend((float(p[0].item()), float(p[1].item())) for p in agent)
        for x, y in points:
            if x < x0 or x > x1 or y < y0 or y > y1:
                return False, f"stay_in_arena violated at ({x:.2f},{y:.2f})"
        return True, "stay_in_arena ok"


class NoEnterRestricted(ConstraintRule):
    def __init__(self) -> None:
        super().__init__("no_enter_restricted_zone", ConstraintClass.HARD, "hard no-go zones")

    def evaluate(self, candidate, trajectory, simulation):
        points = [candidate.target_xy]
        if trajectory is not None:
            agent = trajectory.states_xy[:, AGENT_SLOT]
            points.extend((float(p[0].item()), float(p[1].item())) for p in agent)
        for zone in simulation.restricted_zones:
            for x, y in points:
                if zone.x0 <= x <= zone.x1 and zone.y0 <= y <= zone.y1:
                    return False, f"no_enter_restricted_zone at ({x:.2f},{y:.2f}) zone={zone}"
        return True, "no_enter_restricted_zone ok"


class NoCollideObstacle(ConstraintRule):
    def __init__(self) -> None:
        super().__init__("no_collide_obstacle", ConstraintClass.HARD, "do not plan through obstacles")

    def evaluate(self, candidate, trajectory, simulation):
        for obs in simulation.obstacles:
            ox, oy = float(obs["xy"][0]), float(obs["xy"][1])
            sx, sy = float(obs.get("size", [1.0, 1.0])[0]), float(obs.get("size", [1.0, 1.0])[1])
            x0, x1 = ox - sx / 2, ox + sx / 2
            y0, y1 = oy - sy / 2, oy + sy / 2
            tx, ty = candidate.target_xy
            if x0 <= tx <= x1 and y0 <= ty <= y1:
                return False, f"no_collide_obstacle target inside obstacle {obs.get('id')}"
        return True, "no_collide_obstacle ok"


class MaxSpeed(ConstraintRule):
    def __init__(self) -> None:
        super().__init__("max_speed", ConstraintClass.HARD, "commanded speed may not exceed platform limit")

    def evaluate(self, candidate, trajectory, simulation):
        speed = float(candidate.parameters.get("speed", 1.0))
        if speed > simulation.max_speed + 1e-6:
            return False, f"max_speed {speed} > {simulation.max_speed}"
        return True, "max_speed ok"


RULE_REGISTRY: dict[str, type[ConstraintRule]] = {
    "stay_in_arena": StayInArena,
    "no_enter_restricted_zone": NoEnterRestricted,
    "no_collide_obstacle": NoCollideObstacle,
    "max_speed": MaxSpeed,
}
