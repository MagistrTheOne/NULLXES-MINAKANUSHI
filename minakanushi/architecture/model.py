"""MINAKANUSHI system module: perception + NPF + DWC + memory + future."""

from __future__ import annotations

from torch import Tensor, nn

from minakanushi.architecture.config import ArchitectureConfig
from minakanushi.architecture.mina_unit import MinaUnitBatch
from minakanushi.architecture.outputs import CoreOutput, PositionState
from minakanushi.core.dynamic_world_core import DynamicWorldCore
from minakanushi.future.engine import FutureEngine
from minakanushi.memory.engine import MemoryEngine
from minakanushi.perception.bridge import PerceptionBridge
from minakanushi.position.field import NullxesPositionField
from minakanushi.state.world import WorldState
from minakanushi.uncertainty.engine import UncertaintyEngine


class MinakanushiSystem(nn.Module):
    """Learned MINAKANUSHI substrate. Constraint kernel is not a learned module."""

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        self.perception = PerceptionBridge(config)
        self.position_field = NullxesPositionField(config)
        self.world_core = DynamicWorldCore(config)
        self.uncertainty = UncertaintyEngine(config)
        self.memory = MemoryEngine(config)
        self.future = FutureEngine(config)

    def position_units(self, units: MinaUnitBatch) -> PositionState:
        return self.position_field(
            sequence_position=units.sequence_index,
            timestamp=units.timestamp,
            spatial_position=units.spatial_position,
            episode_position=units.episode_position,
            memory_age=units.memory_age,
            source_id=units.source_id,
            spatial_valid=units.spatial_valid,
            arrival_time=units.arrival_time,
            source_rate=units.source_rate,
        )

    def observe_to_core(
        self,
        units: MinaUnitBatch,
        world: WorldState,
        memory_state: Tensor,
    ) -> tuple[PositionState, CoreOutput]:
        positioned = self.position_units(units)
        core = self.world_core(
            world_state=world,
            observation_state=units.semantic_embedding,
            memory_state=memory_state,
            position_state=positioned,
            units=units,
        )
        self.uncertainty(core.world_state, units)
        return positioned, core

    def parameter_report(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": int(total), "trainable": int(trainable)}
