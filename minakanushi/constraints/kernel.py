"""MinakanushiConstraintKernel (MCK).

Ordering is mandatory and structural:

    StrategyCandidate → kernel → AllowedStrategy → ActionPolicy

ActionPolicy cannot accept raw StrategyCandidate on the production path.
"""

from __future__ import annotations

from minakanushi.architecture.config import SimulationConfig
from minakanushi.constraints.allowed import AllowedStrategy, mint_allowed
from minakanushi.constraints.audit import ConstraintAudit
from minakanushi.constraints.rule import RULE_REGISTRY, ConstraintClass
from minakanushi.future.trajectory import FutureTrajectory
from minakanushi.strategy.candidate import StrategyCandidate


class MinakanushiConstraintKernel:
    def __init__(self, simulation: SimulationConfig) -> None:
        self.simulation = simulation
        self.rules = tuple(RULE_REGISTRY[name]() for name in simulation.hard_constraints if name in RULE_REGISTRY)

    def filter(
        self,
        candidates: list[StrategyCandidate],
        trajectories: dict[str, list[FutureTrajectory]],
    ) -> tuple[tuple[AllowedStrategy, ...], tuple[StrategyCandidate, ...], tuple[ConstraintAudit, ...]]:
        allowed: list[AllowedStrategy] = []
        rejected: list[StrategyCandidate] = []
        audits: list[ConstraintAudit] = []
        for candidate in candidates:
            branches = trajectories.get(candidate.strategy_id, [])
            reasons: list[str] = []
            ok = True
            to_check = branches if branches else [None]
            for traj in to_check:
                for rule in self.rules:
                    passed, reason = rule.evaluate(candidate, traj, self.simulation)
                    reasons.append(reason)
                    if rule.cls == ConstraintClass.HARD and not passed:
                        ok = False
            candidate.constraint_status = "allowed" if ok else "rejected"
            audits.append(ConstraintAudit(candidate.strategy_id, ok, tuple(reasons)))
            if ok:
                allowed.append(mint_allowed(candidate))
            else:
                rejected.append(candidate)
        return tuple(allowed), tuple(rejected), tuple(audits)
