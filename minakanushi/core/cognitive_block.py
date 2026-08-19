"""Slot-relational cognitive block.

Justification: world hypotheses are persistent entities with relations, not a
token stream. Slots query evidence (observation matching) and other slots
(relational structure), then persist via a learned gate. This is not a
decoder-only transformer and does not use causal token attention.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from minakanushi.architecture.config import ArchitectureConfig
from minakanushi.utils.tensors import assert_finite, assert_shape


class CognitiveBlock(nn.Module):
    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        dim = config.latent_dim
        self.dim = dim
        self.obs_norm = nn.LayerNorm(dim)
        self.slot_norm = nn.LayerNorm(dim)
        self.q_obs = nn.Linear(dim, dim)
        self.k_obs = nn.Linear(dim, dim)
        self.v_obs = nn.Linear(dim, dim)
        self.q_rel = nn.Linear(dim, dim)
        self.k_rel = nn.Linear(dim, dim)
        self.v_rel = nn.Linear(dim, dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )
        self.gate = nn.Linear(dim * 2, dim)
        self.drop = nn.Dropout(config.dropout)

    def forward(
        self,
        world_latent: Tensor,
        observation_latent: Tensor,
        occupied: Tensor,
        observation_mask: Tensor,
    ) -> Tensor:
        """
        world_latent:        [B, N_world, D]
        observation_latent:  [B, N_obs, D]
        occupied:            [B, N_world]
        observation_mask:    [B, N_obs]
        """
        batch, n_world, dim = world_latent.shape
        assert_shape("observation_latent", observation_latent, (batch, -1, dim))
        slots = self.slot_norm(world_latent)
        obs = self.obs_norm(observation_latent)
        evidence = self._masked_attention(
            self.q_obs(slots),
            self.k_obs(obs),
            self.v_obs(obs),
            key_mask=observation_mask,
        )
        relations = self._masked_attention(
            self.q_rel(slots),
            self.k_rel(slots),
            self.v_rel(slots),
            key_mask=occupied,
        )
        update = self.drop(self.ff(slots + evidence + relations))
        keep = torch.sigmoid(self.gate(torch.cat([world_latent, update], dim=-1)))
        nxt = keep * world_latent + (1.0 - keep) * (world_latent + update)
        nxt = torch.where(occupied.unsqueeze(-1), nxt, world_latent)
        assert_finite("cognitive_block.output", nxt)
        return nxt

    def _masked_attention(self, query: Tensor, key: Tensor, value: Tensor, key_mask: Tensor) -> Tensor:
        scale = self.dim ** -0.5
        logits = torch.matmul(query, key.transpose(-1, -2)) * scale
        invalid = ~key_mask.unsqueeze(1)
        logits = logits.masked_fill(invalid, -1e9)
        empty = (~key_mask).all(dim=-1, keepdim=True).unsqueeze(-1)
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(empty, torch.zeros_like(weights), weights)
        return torch.matmul(weights, value)
