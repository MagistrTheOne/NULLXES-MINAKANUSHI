"""Constraint and closed-loop metrics must be measured, not hardcoded."""

from __future__ import annotations

from helpers import cpu_config
from minakanushi.constraints.kernel import MinakanushiConstraintKernel
from minakanushi.policy.action_policy import ActionPolicy
from minakanushi.policy.intent import ActionIntent
from minakanushi.strategy.candidate import StrategyCandidate
from minakanushi.training.metrics import (
    closed_loop_success,
    count_hard_violations,
    policy_firewall_metrics,
)


def test_hard_violation_count_is_not_always_zero() -> None:
    sim = cpu_config().simulation
    banned = StrategyCandidate("raid_restricted", "MOVE_TO", (8.2, 8.2), 100.0, 0.0)
    hold = StrategyCandidate("hold", "SAFE_HOLD", (1.0, 1.0), -10.0, 0.0)
    assert count_hard_violations(banned, (), sim) > 0
    assert count_hard_violations(hold, (), sim) == 0


def test_closed_loop_fails_when_executed_command_violates_hard_rule() -> None:
    sim = cpu_config().simulation
    wait = ActionIntent("wait", "WAIT", (1.0, 1.0), {}, 1.0, 1.0, (), "test")
    raid = ActionIntent("raid_restricted", "MOVE_TO", (8.2, 8.2), {}, 1.0, 1.0, (), "test")
    assert closed_loop_success(sim, wait, (1.0, 1.0), (0.0, 0.0), 0.0) == 1.0
    assert closed_loop_success(sim, raid, (1.0, 1.0), (0.0, 0.0), 0.0) == 0.0


def test_policy_firewall_metrics_use_kernel_not_placeholder() -> None:
    sim = cpu_config().simulation
    kernel = MinakanushiConstraintKernel(sim)
    policy = ActionPolicy()
    banned = StrategyCandidate("raid_restricted", "MOVE_TO", (8.2, 8.2), 100.0, 0.0)
    hold = StrategyCandidate("hold", "SAFE_HOLD", (1.0, 1.0), -10.0, 0.0)
    violations, success = policy_firewall_metrics(
        kernel,
        policy,
        [banned, hold],
        {},
        sim,
        sim.home,
        0.0,
        (1.0, 1.0),
        (0.0, 0.0),
    )
    assert violations == 0
    assert success == 1.0
    raw_banned = closed_loop_success(
        sim,
        ActionIntent("raid_restricted", "MOVE_TO", (8.2, 8.2), {}, 1.0, 1.0, (), "bypass"),
        (1.0, 1.0),
        (0.0, 0.0),
        0.0,
    )
    assert raw_banned == 0.0
