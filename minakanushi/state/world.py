"""Self-state and world-entity contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class SelfState:
    platform_id: str
    platform_type: str
    position: tuple[float, float, float]
    orientation: float
    velocity: tuple[float, float, float]
    available_resources: dict[str, float]
    sensor_state: dict[str, float]
    subsystem_state: dict[str, str]
    current_action: str
    operational_limits: dict[str, float]
    health: float


@dataclass
class Entity:
    entity_id: int
    kind: str
    xy: tuple[float, float]
    velocity: tuple[float, float]
    confidence: float
    uncertainty: float
    last_seen: float
    occupied: bool
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class WorldState:
    """Persistent belief about reality, not a copy of the current observation.

    latent_state:     [B, N_world, D]  persistent latent hypotheses
    entity_xy:        [B, N_world, 2]  hypothesized planar position
    entity_vel:       [B, N_world, 2]  hypothesized velocity
    occupied:         [B, N_world]     bool slot occupancy
    entity_id:        [B, N_world]     long, 0 = empty
    kind:             [B, N_world]     long
    confidence:       [B, N_world]
    uncertainty:      [B, N_world, U]
    age_unobserved:   [B, N_world]     steps since last evidence
    timestamp:        [B]
    """

    timestamp: Tensor
    latent_state: Tensor
    entity_xy: Tensor
    entity_vel: Tensor
    occupied: Tensor
    entity_id: Tensor
    kind: Tensor
    confidence: Tensor
    uncertainty: Tensor
    age_unobserved: Tensor
    self_index: int = 0
    provenance: str = "state_constructor"
    corrections: tuple = ()

    @property
    def entity_count(self) -> int:
        return int(self.occupied[0].sum().item()) if self.occupied.ndim == 2 else int(self.occupied.sum().item())
