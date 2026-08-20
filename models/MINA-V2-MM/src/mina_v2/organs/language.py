"""Language organ: operator text → intent slots → MinaUnit(text). Not an LLM. Not ActionPolicy."""

from __future__ import annotations

import torch
from torch import nn

from minakanushi.architecture.config import ArchitectureConfig
from minakanushi.architecture.mina_unit import MinaUnit
from mina_v2.encoder import OrganEncoder
from mina_v2.observation import OperatorIntent

FEATURE_DIM = 6
SOURCE_ID = 12

_PHRASES: dict[str, OperatorIntent] = {
    "найди машину возле склада": OperatorIntent(
        target="vehicle",
        constraint="warehouse",
        priority="search",
        utterance="Найди машину возле склада",
    ),
}


def parse_operator(raw: OperatorIntent | str | None) -> OperatorIntent | None:
    if raw is None:
        return None
    if isinstance(raw, OperatorIntent):
        return raw
    key = " ".join(str(raw).strip().lower().split())
    intent = _PHRASES.get(key)
    if intent is None:
        raise ValueError(f"language adapter has no mapping for {raw!r}")
    return intent


class LanguageAdapter(nn.Module):
    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.encoder = OrganEncoder(FEATURE_DIM, config)

    def encode(
        self,
        operator: OperatorIntent | str | None,
        *,
        timestamp: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[MinaUnit]:
        intent = parse_operator(operator)
        if intent is None:
            return []
        feat = torch.tensor(
            [
                float(hash(intent.target) % 997) / 997.0,
                float(hash(intent.constraint) % 997) / 997.0,
                float(hash(intent.priority) % 997) / 997.0,
                1.0,
                0.0,
                0.0,
            ],
            device=device,
            dtype=dtype,
        )
        return [
            MinaUnit(
                source_type="text",
                source_id=SOURCE_ID,
                timestamp=timestamp,
                sequence_index=1,
                spatial_frame="arena",
                spatial_position=(0.0, 0.0, 0.0),
                spatial_valid=False,
                semantic_embedding=self.encoder(feat.unsqueeze(0)).squeeze(0),
                confidence=1.0,
                uncertainty=0.0,
                persistence=0.9,
                entity_reference=0,
                relation_reference=0,
                kind="event",
                arrival_time=timestamp,
                metadata={
                    "organ": "language",
                    "target": intent.target,
                    "constraint": intent.constraint,
                    "priority": intent.priority,
                    "utterance": intent.utterance or "",
                    "action_intent": False,
                },
            )
        ]
