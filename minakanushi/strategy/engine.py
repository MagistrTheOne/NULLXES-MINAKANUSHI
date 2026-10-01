"""Strategy Engine — generate multiple candidates before selection."""

from __future__ import annotations

from minakanushi.architecture.mina_unit import KIND_IDS
from minakanushi.situation.core import SituationState
from minakanushi.state.entity import AGENT_SLOT
from minakanushi.strategy.candidate import StrategyCandidate


class StrategyEngine:
    VOCABULARY = (
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

    def _risk(self, target: tuple[float, float], world) -> float:
        """Proximity risk: 1 inside an obstacle slot radius, decaying to 0 at 1.5m.

        Obstacle slots carry no size in WorldState, so a fixed 0.5m footprint
        radius is assumed (matches AGENT_MARGIN + half body in rule.py).
        Deterministic; no learned weights.
        """
        risk = 0.0
        for slot in world.occupied[0].nonzero(as_tuple=False).flatten().tolist():
            if int(world.kind[0, slot].item()) != KIND_IDS.get("obstacle", -1):
                continue
            ox = float(world.entity_xy[0, slot, 0].item())
            oy = float(world.entity_xy[0, slot, 1].item())
            dist = ((target[0] - ox) ** 2 + (target[1] - oy) ** 2) ** 0.5
            risk = max(risk, max(0.0, 1.0 - dist / 1.5))
        return risk

    def generate(self, situation: SituationState, home: tuple[float, float]) -> list[StrategyCandidate]:
        world = situation.world_state
        agent_xy = (
            float(world.entity_xy[0, AGENT_SLOT, 0].item()),
            float(world.entity_xy[0, AGENT_SLOT, 1].item()),
        )
        candidates = [
            StrategyCandidate("observe", "OBSERVE", agent_xy, expected_value=-0.4, uncertainty=situation.uncertainty,
                              parameters={"speed": 0.0}),
            StrategyCandidate("wait", "WAIT", agent_xy, expected_value=-0.5, uncertainty=situation.uncertainty,
                              parameters={"speed": 0.0}),
            StrategyCandidate("safe_hold", "SAFE_HOLD", agent_xy, expected_value=-0.2, uncertainty=0.1,
                              parameters={"speed": 0.0}),
            StrategyCandidate("abort", "ABORT", agent_xy, expected_value=-1.0, uncertainty=0.1,
                              parameters={"speed": 0.0}),
            StrategyCandidate("return", "RETURN", home, expected_value=-self._dist(agent_xy, home), uncertainty=0.2,
                              parameters={"speed": 1.0}),
            StrategyCandidate(
                "request_assistance",
                "REQUEST_ASSISTANCE",
                agent_xy,
                expected_value=-0.8,
                uncertainty=max(0.3, situation.uncertainty),
                parameters={"speed": 0.0},
            ),
        ]
        for slot in world.occupied[0].nonzero(as_tuple=False).flatten().tolist():
            kind = int(world.kind[0, slot].item())
            eid = int(world.entity_id[0, slot].item())
            xy = (
                float(world.entity_xy[0, slot, 0].item()),
                float(world.entity_xy[0, slot, 1].item()),
            )
            if kind == KIND_IDS["target"]:
                candidates.append(
                    StrategyCandidate(
                        f"move_to_{eid}",
                        "MOVE_TO",
                        xy,
                        expected_value=-self._dist(agent_xy, xy),
                        uncertainty=float(world.uncertainty[0, slot].mean().item()),
                        predicted_risk=self._risk(xy, world),
                        parameters={"entity_id": float(eid), "speed": 1.0},
                    )
                )
            if kind == KIND_IDS["mover"]:
                speed_mag = float((world.entity_vel[0, slot] ** 2).sum().sqrt().item())
                mover_risk = min(1.0, self._risk(xy, world) + 0.2 * min(1.0, speed_mag))
                candidates.append(
                    StrategyCandidate(
                        f"follow_{eid}",
                        "FOLLOW",
                        xy,
                        expected_value=-self._dist(agent_xy, xy) - 0.3,
                        uncertainty=float(world.uncertainty[0, slot].mean().item()),
                        predicted_risk=mover_risk,
                        parameters={"entity_id": float(eid), "speed": 1.0},
                    )
                )
                candidates.append(
                    StrategyCandidate(
                        f"inspect_{eid}",
                        "INSPECT",
                        xy,
                        expected_value=-self._dist(agent_xy, xy) - 0.1,
                        uncertainty=float(world.uncertainty[0, slot].mean().item()),
                        predicted_risk=mover_risk,
                        parameters={"entity_id": float(eid), "speed": 1.0},
                    )
                )
        return candidates

    def _dist(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
