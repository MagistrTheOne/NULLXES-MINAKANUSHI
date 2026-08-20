"""Audio organ: evidence events → MinaUnit(structured_event). Not audio generation."""

from __future__ import annotations

import torch
from torch import nn

from minakanushi.architecture.config import ArchitectureConfig
from minakanushi.architecture.mina_unit import MinaUnit
from mina_v2.encoder import OrganEncoder
from mina_v2.observation import AUDIO_KINDS, AudioEvent

FEATURE_DIM = 4
SOURCE_ID = 11


class AudioAdapter(nn.Module):
    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.encoder = OrganEncoder(FEATURE_DIM, config)

    def encode(
        self,
        events: tuple[AudioEvent, ...],
        *,
        timestamp: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[MinaUnit]:
        units: list[MinaUnit] = []
        for seq, event in enumerate(events, start=1):
            conf = float(event.confidence)
            idx = AUDIO_KINDS.index(event.kind)
            feat = torch.tensor(
                [float(idx), conf, 1.0, 0.0],
                device=device,
                dtype=dtype,
            )
            units.append(
                MinaUnit(
                    source_type="structured_event",
                    source_id=SOURCE_ID,
                    timestamp=timestamp,
                    sequence_index=seq,
                    spatial_frame="arena",
                    spatial_position=(0.0, 0.0, 0.0),
                    spatial_valid=False,
                    semantic_embedding=self.encoder(feat.unsqueeze(0)).squeeze(0),
                    confidence=conf,
                    uncertainty=max(0.0, 1.0 - conf),
                    persistence=0.5,
                    entity_reference=0,
                    relation_reference=0,
                    kind="event",
                    arrival_time=timestamp,
                    metadata={"channel": "audio", "kind": event.kind, "organ": "audio"},
                )
            )
        return units
