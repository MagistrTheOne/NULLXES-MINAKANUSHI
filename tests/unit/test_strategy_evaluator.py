"""Strategy value evaluation should not warn on live gradient tensors."""

from __future__ import annotations

import warnings

import torch

from minakanushi.future.trajectory import FutureTrajectory
from minakanushi.strategy.candidate import StrategyCandidate
from minakanushi.strategy.evaluator import evaluate_value


def test_evaluate_value_detaches_uncertainty_scalar() -> None:
    candidate = StrategyCandidate(
        strategy_id="move",
        objective="MOVE_TO",
        target_xy=(1.0, 1.0),
        expected_value=0.0,
        uncertainty=0.1,
        predicted_risk=0.2,
    )
    terminal = torch.zeros(16, 2)
    trajectory = FutureTrajectory(
        states_xy=torch.zeros(2, 16, 2),
        probability=torch.tensor(1.0),
        uncertainty=torch.tensor(0.5, requires_grad=True),
        causal_assumptions=(),
        terminal_xy=terminal,
        action_id="move",
        strategy_id="move",
        branch_id=0,
        horizon_steps=2,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = evaluate_value(candidate, trajectory, (1.0, 1.0))
    assert value < 0.0
    assert not any("requires_grad=True to a scalar" in str(item.message) for item in caught)
