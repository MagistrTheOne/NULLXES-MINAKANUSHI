"""AllowedStrategy can only be minted by MinakanushiConstraintKernel."""

from __future__ import annotations

from dataclasses import dataclass

from minakanushi.strategy.candidate import StrategyCandidate


_KERNEL_TOKEN = object()


@dataclass(frozen=True)
class AllowedStrategy:
    candidate: StrategyCandidate
    _token: object

    def __post_init__(self) -> None:
        if self._token is not _KERNEL_TOKEN:
            raise TypeError("AllowedStrategy can only be constructed by MinakanushiConstraintKernel")

    @property
    def strategy_id(self) -> str:
        return self.candidate.strategy_id

    @property
    def objective(self) -> str:
        return self.candidate.objective

    @property
    def target_xy(self) -> tuple[float, float]:
        return self.candidate.target_xy


def mint_allowed(candidate: StrategyCandidate) -> AllowedStrategy:
    return AllowedStrategy(candidate, _KERNEL_TOKEN)
