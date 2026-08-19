"""NullxesPositionField (NPF).

Native position is multidimensional:

    P_i = (p_seq, p_time, p_space, p_episode, p_memory, p_source)

Physical time is not token index. This module is not RoPE and is not aliased
to another architecture's positional system.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from minakanushi.architecture.config import ArchitectureConfig
from minakanushi.architecture.outputs import PositionState
from minakanushi.position.episode import EpisodeEncoder
from minakanushi.position.memory_age import MemoryAgeEncoder
from minakanushi.position.sequence import SequenceEncoder
from minakanushi.position.source import SourceEncoder
from minakanushi.position.spatial import SpatialEncoder
from minakanushi.position.temporal import PhysicalTimeEncoder
from minakanushi.utils.tensors import assert_finite, assert_shape


class NullxesPositionField(nn.Module):
    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        dim = config.latent_dim
        freqs = config.npf.num_frequencies
        self.config = config
        self.sequence_encoder = SequenceEncoder(dim, freqs)
        self.time_encoder = PhysicalTimeEncoder(dim, freqs, scale=config.npf.time_scale_seconds)
        self.space_encoder = SpatialEncoder(dim)
        self.episode_encoder = EpisodeEncoder(dim, freqs)
        self.memory_encoder = MemoryAgeEncoder(dim, freqs)
        self.source_encoder = SourceEncoder(dim, config.npf.max_sources)
        self.mixer = nn.Sequential(
            nn.Linear(dim * 6, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.gate = nn.Linear(dim * 6, 6)

    def forward(
        self,
        sequence_position: Tensor,
        timestamp: Tensor,
        spatial_position: Tensor,
        episode_position: Tensor,
        memory_age: Tensor,
        source_id: Tensor,
        spatial_valid: Tensor | None = None,
        arrival_time: Tensor | None = None,
        source_rate: Tensor | None = None,
    ) -> PositionState:
        """timestamp is event_time [B, N]. arrival_time / source_rate required for async sensors."""
        batch, count = timestamp.shape
        dim = self.config.latent_dim
        if spatial_valid is None:
            spatial_valid = torch.ones(batch, count, dtype=torch.bool, device=timestamp.device)
        if arrival_time is None or source_rate is None:
            raise ValueError("NPF requires arrival_time and source_rate; event_time==arrival_time must be explicit, not implicit")
        sequence_embedding = self.sequence_encoder(sequence_position)
        temporal_embedding = self.time_encoder(timestamp, arrival_time, source_rate)
        spatial_embedding = self.space_encoder(spatial_position, spatial_valid)
        episode_embedding = self.episode_encoder(episode_position)
        memory_embedding = self.memory_encoder(memory_age)
        source_embedding = self.source_encoder(source_id)
        stacked = torch.cat(
            [
                sequence_embedding,
                temporal_embedding,
                spatial_embedding,
                episode_embedding,
                memory_embedding,
                source_embedding,
            ],
            dim=-1,
        )
        gates = torch.softmax(self.gate(stacked), dim=-1).unsqueeze(-1)
        parts = torch.stack(
            [
                sequence_embedding,
                temporal_embedding,
                spatial_embedding,
                episode_embedding,
                memory_embedding,
                source_embedding,
            ],
            dim=-2,
        )
        gated = (parts * gates).sum(dim=-2)
        mixed = self.mixer(stacked) + gated
        assert_shape("PositionState.embedding", mixed, (batch, count, dim))
        assert_finite("PositionState.embedding", mixed)
        return PositionState(
            embedding=mixed,
            temporal_embedding=temporal_embedding,
            spatial_embedding=spatial_embedding,
            episode_embedding=episode_embedding,
            memory_embedding=memory_embedding,
            source_embedding=source_embedding,
            sequence_embedding=sequence_embedding,
        )
