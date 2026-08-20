"""Position test: identical observations at different physical times differ in NPF."""

from __future__ import annotations

import torch

from minakanushi.architecture.config import load_architecture
from minakanushi.architecture.mina_unit import MinaUnit
from minakanushi.architecture.mina_unit import pack_units
from minakanushi.position.field import NullxesPositionField
from helpers import ROOT


def test_physical_time_changes_position_state() -> None:
    config = load_architecture(ROOT / "configs" / "architecture" / "cpu_dev.yaml")
    npf = NullxesPositionField(config)
    embedding = torch.zeros(config.latent_dim)
    unit_a = MinaUnit(
        source_type="vector",
        source_id=2,
        timestamp=1.0,
        sequence_index=0,
        spatial_frame="arena",
        spatial_position=(3.0, 4.0, 0.0),
        spatial_valid=True,
        semantic_embedding=embedding,
        confidence=1.0,
        uncertainty=0.0,
        persistence=1.0,
        entity_reference=11,
        relation_reference=0,
        kind="mover",
    )
    unit_b = MinaUnit(
        source_type="vector",
        source_id=2,
        timestamp=5.0,
        sequence_index=0,
        spatial_frame="arena",
        spatial_position=(3.0, 4.0, 0.0),
        spatial_valid=True,
        semantic_embedding=embedding.clone(),
        confidence=1.0,
        uncertainty=0.0,
        persistence=1.0,
        entity_reference=11,
        relation_reference=0,
        kind="mover",
    )
    a = pack_units([unit_a], batch_index=0, max_units=4, latent_dim=config.latent_dim, episode_position=0.0, now=5.0, device=torch.device("cpu"), dtype=torch.float32)
    b = pack_units([unit_b], batch_index=0, max_units=4, latent_dim=config.latent_dim, episode_position=0.0, now=5.0, device=torch.device("cpu"), dtype=torch.float32)
    pa = npf(
        a.sequence_index,
        a.timestamp,
        a.spatial_position,
        a.episode_position,
        a.memory_age,
        a.source_id,
        a.spatial_valid,
        arrival_time=a.arrival_time,
        source_rate=a.source_rate,
    )
    pb = npf(
        b.sequence_index,
        b.timestamp,
        b.spatial_position,
        b.episode_position,
        b.memory_age,
        b.source_id,
        b.spatial_valid,
        arrival_time=b.arrival_time,
        source_rate=b.source_rate,
    )
    assert not torch.allclose(pa.temporal_embedding[0, 0], pb.temporal_embedding[0, 0])
    assert not torch.allclose(pa.embedding[0, 0], pb.embedding[0, 0])
