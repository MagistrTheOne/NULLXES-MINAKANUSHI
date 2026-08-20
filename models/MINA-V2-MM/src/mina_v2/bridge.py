"""V2 Perception Bridge — organs into MinaUnit Space. Does not replace 1.0 PerceptionBridge."""

from __future__ import annotations

import torch
from torch import nn

from minakanushi.architecture.config import ArchitectureConfig
from minakanushi.architecture.mina_unit import MinaUnit
from minakanushi.perception.bridge import PerceptionBridge
from mina_v2.observation import MultimodalObservation
from mina_v2.organs.audio import AudioAdapter
from mina_v2.organs.language import LanguageAdapter
from mina_v2.organs.sensor import SensorAdapter
from mina_v2.organs.vision import VisionAdapter


class V2PerceptionBridge(nn.Module):
    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        self.foundation = PerceptionBridge(config)
        self.vision = VisionAdapter(config)
        self.audio = AudioAdapter(config)
        self.language = LanguageAdapter(config)
        self.sensor = SensorAdapter(config)

    def encode(
        self,
        observation: MultimodalObservation,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[MinaUnit]:
        units = list(self.foundation.encode(observation.base, device=device, dtype=dtype))
        now = float(observation.base.timestamp)
        units.extend(self.vision.encode(observation.detections, timestamp=now, device=device, dtype=dtype))
        units.extend(self.audio.encode(observation.audio, timestamp=now, device=device, dtype=dtype))
        units.extend(self.language.encode(observation.operator, timestamp=now, device=device, dtype=dtype))
        units.extend(self.sensor.encode(observation.embodiment, timestamp=now, device=device, dtype=dtype))
        for unit in units:
            unit.validate()
        return units
