"""event_time and arrival_time must produce distinct NPF states when they differ."""

from __future__ import annotations

import torch

from minakanushi.architecture.config import load_architecture
from minakanushi.architecture.mina_unit import MinaUnit, pack_units
from minakanushi.position.field import NullxesPositionField
from helpers import ROOT


def test_event_time_is_not_arrival_time() -> None:
    config = load_architecture(ROOT / "configs" / "architecture" / "cpu_dev.yaml")
    npf = NullxesPositionField(config)
    emb = torch.zeros(config.latent_dim)

    def pack(event: float, arrival: float):
        unit = MinaUnit(
            source_type="vector",
            source_id=2,
            timestamp=event,
            sequence_index=0,
            spatial_frame="arena",
            spatial_position=(3.0, 4.0, 0.0),
            spatial_valid=True,
            semantic_embedding=emb.clone(),
            confidence=1.0,
            uncertainty=0.0,
            persistence=1.0,
            entity_reference=11,
            relation_reference=0,
            kind="mover",
            arrival_time=arrival,
            source_rate=10.0,
        )
        return pack_units(
            [unit],
            batch_index=0,
            max_units=4,
            latent_dim=config.latent_dim,
            episode_position=0.0,
            now=arrival,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    synced = pack(1.0, 1.0)
    delayed = pack(1.0, 1.4)
    p0 = npf(
        synced.sequence_index,
        synced.timestamp,
        synced.spatial_position,
        synced.episode_position,
        synced.memory_age,
        synced.source_id,
        synced.spatial_valid,
        arrival_time=synced.arrival_time,
        source_rate=synced.source_rate,
    )
    p1 = npf(
        delayed.sequence_index,
        delayed.timestamp,
        delayed.spatial_position,
        delayed.episode_position,
        delayed.memory_age,
        delayed.source_id,
        delayed.spatial_valid,
        arrival_time=delayed.arrival_time,
        source_rate=delayed.source_rate,
    )
    assert float((synced.timestamp - delayed.timestamp).abs().sum()) == 0.0
    assert not torch.allclose(p0.temporal_embedding[0, 0], p1.temporal_embedding[0, 0])
