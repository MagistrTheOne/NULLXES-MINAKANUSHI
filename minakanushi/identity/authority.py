"""AuthorityModel — decision permission. Not cognition on/off."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

from minakanushi.constraints.allowed import AllowedStrategy
from minakanushi.future.trajectory import FutureTrajectory
from minakanushi.policy.action_policy import ActionPolicy
from minakanushi.policy.intent import ActionIntent
from minakanushi.strategy.candidate import StrategyCandidate


class AuthorityMode(str, Enum):
    AUTONOMOUS = "AUTONOMOUS"
    ADVISORY = "ADVISORY"
    DIRECTED = "DIRECTED"
    MANUAL = "MANUAL"
    SAFE_HOLD = "SAFE_HOLD"


COGNITION_ALWAYS_ON = (
    "world_state",
    "memory",
    "uncertainty",
    "future",
    "situation",
    "constraint_kernel",
    "telemetry",
)


def _hold(goal_xy: tuple[float, float], now: float, reason: str) -> ActionIntent:
    return ActionIntent(
        strategy_id="safe_hold",
        objective="SAFE_HOLD",
        target_state=goal_xy,
        parameters={},
        confidence=1.0,
        valid_until=now + 1.0,
        abort_conditions=("authority_hold",),
        provenance=reason,
    )


@dataclass
class AuthorityModel:
    mode: AuthorityMode = AuthorityMode.AUTONOMOUS
    policy_enabled: bool = True
    operator_connected: bool = False

    def to_dict(self) -> dict:
        return {"mode": self.mode.value, "policy_enabled": self.policy_enabled, "operator_connected": self.operator_connected}

    @classmethod
    def from_dict(cls, raw: dict) -> AuthorityModel:
        mode = raw.get("mode", AuthorityMode.AUTONOMOUS.value)
        return cls(
            mode=AuthorityMode(mode),
            policy_enabled=bool(raw.get("policy_enabled", True)),
            operator_connected=bool(raw.get("operator_connected", False)),
        )

    def selection_enabled(self) -> bool:
        if not self.policy_enabled:
            return False
        return self.mode == AuthorityMode.AUTONOMOUS

    def resolve(
        self,
        policy: ActionPolicy,
        allowed: tuple[AllowedStrategy, ...] | list[AllowedStrategy],
        trajectories: dict[str, list[FutureTrajectory]],
        goal_xy: tuple[float, float],
        now: float,
        operator_intent: ActionIntent | None = None,
    ) -> ActionIntent:
        if self.mode == AuthorityMode.SAFE_HOLD or not self.policy_enabled or self.mode == AuthorityMode.MANUAL:
            return _hold(goal_xy, now, f"authority.{self.mode.value.lower()}.policy_off")
        if self.mode == AuthorityMode.ADVISORY:
            return _hold(goal_xy, now, "authority.advisory.no_autonomous_intent")
        if self.mode == AuthorityMode.DIRECTED:
            if operator_intent is None:
                return _hold(goal_xy, now, "authority.directed.missing_operator_intent")
            allowed_ids = {item.strategy_id for item in allowed}
            if operator_intent.strategy_id not in allowed_ids and operator_intent.objective != "SAFE_HOLD":
                return _hold(goal_xy, now, "authority.directed.kernel_rejected")
            return ActionIntent(
                strategy_id=operator_intent.strategy_id,
                objective=operator_intent.objective,
                target_state=operator_intent.target_state,
                parameters=dict(operator_intent.parameters),
                confidence=operator_intent.confidence,
                valid_until=now + 1.0,
                abort_conditions=operator_intent.abort_conditions + ("hard_constraint_violation",),
                provenance="authority.directed.operator",
            )
        return policy.select(allowed, trajectories, goal_xy, now)


def candidate_from_intent(intent: ActionIntent) -> StrategyCandidate:
    return StrategyCandidate(intent.strategy_id, intent.objective, intent.target_state, 0.0, 0.0)
