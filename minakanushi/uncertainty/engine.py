"""UncertaintyEngine — first-class 'I do not know'.

Channels (U=config.uncertainty_channels), semantics:
  0 missing evidence
  1 noisy evidence
  2 conflicting evidence
  3 out-of-distribution
  4 future ambiguity
  5 strategy uncertainty
  6 state uncertainty
  7 model uncertainty
These are not interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from minakanushi.architecture.config import ArchitectureConfig
from minakanushi.architecture.mina_unit import MinaUnitBatch
from minakanushi.state.world import WorldState
from minakanushi.uncertainty.conflict import conflict_score


@dataclass
class UncertaintyState:
    observation_uncertainty: Tensor
    state_uncertainty: Tensor
    prediction_uncertainty: Tensor
    strategy_uncertainty: Tensor
    model_uncertainty: Tensor
    conflict_score: Tensor
    channels: Tensor


class UncertaintyEngine(nn.Module):
    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        self.proj = nn.Linear(config.latent_dim, config.uncertainty_channels)

    def forward(self, world: WorldState, units: MinaUnitBatch) -> UncertaintyState:
        u = self.config.uncertainty_channels
        learned = torch.sigmoid(self.proj(world.latent_state))
        missing = (world.age_unobserved > 0).to(learned.dtype)
        noisy = (1.0 - world.confidence).clamp(0.0, 1.0)
        conflict = conflict_score(world, units)
        cols = [learned[..., i] for i in range(u)]
        cols[0] = torch.maximum(cols[0], missing)
        cols[1] = torch.maximum(cols[1], noisy)
        cols[2] = torch.maximum(cols[2], conflict)
        cols[2] = torch.maximum(cols[2], world.uncertainty[..., 2])
        cols[6] = torch.maximum(cols[6], world.uncertainty.mean(dim=-1))
        channels = torch.stack(cols, dim=-1)
        occupied = world.occupied.to(channels.dtype).unsqueeze(-1)
        channels = channels * occupied
        world.uncertainty = channels
        mean_u = channels.mean(dim=-1)
        return UncertaintyState(
            observation_uncertainty=noisy,
            state_uncertainty=mean_u,
            prediction_uncertainty=channels[..., 4],
            strategy_uncertainty=channels[..., 5],
            model_uncertainty=channels[..., 7],
            conflict_score=conflict,
            channels=channels,
        )
