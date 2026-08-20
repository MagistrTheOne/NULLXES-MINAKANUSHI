"""Self-state and world-entity contracts.

Public object is Belief (mean + distribution + existence).
latent_state is the learned carrier, not the belief.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor

BELIEF_STD_MIN = 1e-3
BELIEF_EXISTENCE_FLOOR = 5e-2
MEMORY_MEAN_GAIN = 0.1
COAST_STD_GAIN = 0.25
EXISTENCE_DECAY = 0.85
PRED_CONF_DECAY = 0.9


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
class BeliefView:
    """Public belief over world slots. Not a DWC latent rename.

    position_mean / velocity_mean alias WorldState kinematics.
    existence_probability is independent of occupied (slot allocation).
    """

    position_mean: Tensor
    position_std: Tensor
    velocity_mean: Tensor
    velocity_std: Tensor
    existence_probability: Tensor
    last_observation_age: Tensor
    prediction_confidence: Tensor


@dataclass
class WorldState:
    """Persistent belief about reality, not a copy of the current observation.

    latent_state:      [B, N_world, D]  learned carrier, not the belief
    entity_xy:         [B, N_world, 2]  position mean
    xy_std:            [B, N_world, 2]  position std, clamped >= BELIEF_STD_MIN
    entity_vel:        [B, N_world, 2]  velocity mean
    vel_std:           [B, N_world, 2]  velocity std
    existence:         [B, N_world]     P(real) in (0, 1] while occupied
    pred_confidence:   [B, N_world]     confidence in the dynamics prior
    occupied:          [B, N_world]     bool slot occupancy
    entity_id:         [B, N_world]     long, 0 = empty
    kind:              [B, N_world]     long
    confidence:        [B, N_world]
    uncertainty:       [B, N_world, U]
    age_unobserved:    [B, N_world]     steps since last evidence
    timestamp:         [B]
    self_index:        embodiment platform slot in this world, not architecture identity
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
    xy_std: Tensor
    vel_std: Tensor
    existence: Tensor
    pred_confidence: Tensor
    self_index: int = 0
    provenance: str = "state_constructor"
    corrections: tuple = ()

    @property
    def entity_count(self) -> int:
        return int(self.occupied[0].sum().item()) if self.occupied.ndim == 2 else int(self.occupied.sum().item())

    def as_belief(self) -> BeliefView:
        return BeliefView(
            position_mean=self.entity_xy,
            position_std=self.xy_std,
            velocity_mean=self.entity_vel,
            velocity_std=self.vel_std,
            existence_probability=self.existence,
            last_observation_age=self.age_unobserved,
            prediction_confidence=self.pred_confidence,
        )
