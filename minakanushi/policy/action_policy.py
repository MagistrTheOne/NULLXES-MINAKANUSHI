"""Action policy selects only among kernel-minted AllowedStrategy objects."""

from __future__ import annotations

from minakanushi.constraints.allowed import AllowedStrategy
from minakanushi.future.trajectory import FutureTrajectory
from minakanushi.policy.intent import ActionIntent
from minakanushi.strategy.evaluator import evaluate_value


class ActionPolicy:
    def select(
        self,
        allowed: tuple[AllowedStrategy, ...] | list[AllowedStrategy],
        trajectories: dict[str, list[FutureTrajectory]],
        goal_xy: tuple[float, float],
        now: float,
    ) -> ActionIntent:
        if allowed and not isinstance(allowed[0], AllowedStrategy):
            raise TypeError("ActionPolicy.select requires AllowedStrategy from ConstraintKernel, not raw StrategyCandidate")
        if not allowed:
            return ActionIntent(
                strategy_id="safe_hold",
                objective="SAFE_HOLD",
                target_state=goal_xy,
                parameters={},
                confidence=1.0,
                valid_until=now + 1.0,
                abort_conditions=("constraint_kernel_empty_allowed_set",),
                provenance="action_policy.fail_closed",
            )
        scored = []
        for item in allowed:
            branches = trajectories.get(item.strategy_id, [])
            traj = max(branches, key=lambda t: float(t.probability.detach())) if branches else None
            value = evaluate_value(item.candidate, traj, goal_xy)
            item.candidate.expected_value = value
            scored.append((value, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        best = scored[0][1]
        return ActionIntent(
            strategy_id=best.strategy_id,
            objective=best.objective,
            target_state=best.target_xy,
            parameters=dict(best.candidate.parameters),
            confidence=max(0.0, 1.0 - best.candidate.uncertainty),
            valid_until=now + 1.0,
            abort_conditions=("hard_constraint_violation", "health_low"),
            provenance="action_policy.argmax_allowed",
        )
