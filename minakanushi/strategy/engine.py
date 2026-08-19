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

    def generate(self, situation: SituationState, home: tuple[float, float]) -> list[StrategyCandidate]:
        world = situation.world_state
        agent_xy = (
            float(world.entity_xy[0, AGENT_SLOT, 0].item()),
            float(world.entity_xy[0, AGENT_SLOT, 1].item()),
        )
        candidates = [
            StrategyCandidate("observe", "OBSERVE", agent_xy, expected_value=-0.4, uncertainty=situation.uncertainty),
            StrategyCandidate("wait", "WAIT", agent_xy, expected_value=-0.5, uncertainty=situation.uncertainty),
            StrategyCandidate("safe_hold", "SAFE_HOLD", agent_xy, expected_value=-0.2, uncertainty=0.1),
            StrategyCandidate("abort", "ABORT", agent_xy, expected_value=-1.0, uncertainty=0.1),
            StrategyCandidate("return", "RETURN", home, expected_value=-self._dist(agent_xy, home), uncertainty=0.2),
            StrategyCandidate(
                "request_assistance",
                "REQUEST_ASSISTANCE",
                agent_xy,
                expected_value=-0.8,
                uncertainty=max(0.3, situation.uncertainty),
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
                        parameters={"entity_id": float(eid)},
                    )
                )
            if kind == KIND_IDS["mover"]:
                candidates.append(
                    StrategyCandidate(
                        f"follow_{eid}",
                        "FOLLOW",
                        xy,
                        expected_value=-self._dist(agent_xy, xy) - 0.3,
                        uncertainty=float(world.uncertainty[0, slot].mean().item()),
                    )
                )
                candidates.append(
                    StrategyCandidate(
                        f"inspect_{eid}",
                        "INSPECT",
                        xy,
                        expected_value=-self._dist(agent_xy, xy) - 0.1,
                        uncertainty=float(world.uncertainty[0, slot].mean().item()),
                    )
                )
        return candidates

    def _dist(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
