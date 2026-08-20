"""Vision organ: structured detections → MinaUnit(source_type=image). Not a ViT."""

from __future__ import annotations

import torch
from torch import nn

from minakanushi.architecture.config import ArchitectureConfig
from minakanushi.architecture.mina_unit import KIND_IDS, MinaUnit
from mina_v2.encoder import OrganEncoder
from mina_v2.observation import VISION_KIND_MAP, VisionDetection

FEATURE_DIM = 8
SOURCE_ID = 10


def _kind(type_name: str) -> str:
    mapped = VISION_KIND_MAP.get(str(type_name), "unknown")
    if mapped not in KIND_IDS:
        return "unknown"
    return mapped


class VisionAdapter(nn.Module):
    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = OrganEncoder(FEATURE_DIM, config)

    def encode(
        self,
        detections: tuple[VisionDetection, ...],
        *,
        timestamp: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[MinaUnit]:
        units: list[MinaUnit] = []
        for seq, det in enumerate(detections, start=1):
            conf = float(det.confidence)
            kind = _kind(det.type)
            feat = torch.tensor(
                [
                    float(KIND_IDS[kind]),
                    float(det.xy[0]),
                    float(det.xy[1]),
                    float(det.vel[0]),
                    float(det.vel[1]),
                    conf,
                    float(hash(det.relation) % 997) / 997.0,
                    float(hash(det.event) % 997) / 997.0,
                ],
                device=device,
                dtype=dtype,
            )
            units.append(
                MinaUnit(
                    source_type="image",
                    source_id=SOURCE_ID,
                    timestamp=timestamp,
                    sequence_index=seq,
                    spatial_frame="arena",
                    spatial_position=(float(det.xy[0]), float(det.xy[1]), 0.0),
                    spatial_valid=True,
                    semantic_embedding=self.encoder(feat.unsqueeze(0)).squeeze(0),
                    confidence=conf,
                    uncertainty=max(0.0, 1.0 - conf),
                    persistence=0.8,
                    entity_reference=int(det.id),
                    relation_reference=0,
                    kind=kind,
                    arrival_time=timestamp,
                    metadata={
                        "type": det.type,
                        "relation": det.relation,
                        "event": det.event,
                        "vel": det.vel,
                        "organ": "vision",
                    },
                )
            )
        return units
