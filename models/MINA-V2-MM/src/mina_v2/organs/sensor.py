"""Embodiment / physical-state organ → MinaUnit(system_state). Not PWM."""

from __future__ import annotations

import torch
from torch import nn

from minakanushi.architecture.config import ArchitectureConfig
from minakanushi.architecture.mina_unit import MinaUnit
from mina_v2.encoder import OrganEncoder
from mina_v2.observation import EmbodimentState

FEATURE_DIM = 8
SOURCE_ID = 13


class SensorAdapter(nn.Module):
    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.encoder = OrganEncoder(FEATURE_DIM, config)

    def encode(
        self,
        body: EmbodimentState | None,
        *,
        timestamp: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[MinaUnit]:
        if body is None:
            return []
        joints = list(body.joints[:4]) + [0.0] * 4
        feat = torch.tensor(
            [
                float(body.battery),
                float(body.force),
                float(body.balance),
                float(joints[0]),
                float(joints[1]),
                float(joints[2]),
                float(joints[3]),
                0.0,
            ],
            device=device,
            dtype=dtype,
        )
        return [
            MinaUnit(
                source_type="system_state",
                source_id=SOURCE_ID,
                timestamp=timestamp,
                sequence_index=1,
                spatial_frame="arena",
                spatial_position=(0.0, 0.0, 0.0),
                spatial_valid=False,
                semantic_embedding=self.encoder(feat.unsqueeze(0)).squeeze(0),
                confidence=0.98,
                uncertainty=0.02,
                persistence=1.0,
                entity_reference=1,
                relation_reference=0,
                kind="agent",
                arrival_time=timestamp,
                metadata={
                    "organ": "sensor",
                    "battery": float(body.battery),
                    "force": float(body.force),
                    "balance": float(body.balance),
                    "joints": list(body.joints),
                    "pwm": False,
                },
            )
        ]
