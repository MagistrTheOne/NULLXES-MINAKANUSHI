"""Future trajectory contract.

probability: [ ] tensor — P(this branch | strategy), sums to 1 over branches of one strategy
uncertainty: [ ] tensor — reliability of that predicted trajectory; not 1-P
states_xy:   [H, N_world, 2]
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class FutureTrajectory:
    states_xy: Tensor
    probability: Tensor
    uncertainty: Tensor
    causal_assumptions: tuple[str, ...]
    terminal_xy: Tensor
    action_id: str
    strategy_id: str
    branch_id: int
    horizon_steps: int
