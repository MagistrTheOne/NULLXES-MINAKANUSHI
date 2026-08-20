"""MINAKANUSHI configuration contracts.

All latent, slot, and depth values are configuration. Model code must read
these fields rather than assume research_v01 dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class IdentityConfig:
    architecture: str = "MINAKANUSHI"
    organization: str = "NULLXES"
    system_class: str = "adaptive_situational_intelligence"
    architecture_generation: int = 1
    native_runtime: str = "nullxes"
    architecture_version: str = "0.1"

    def validate(self) -> None:
        if self.architecture != "MINAKANUSHI":
            raise ValueError(f"identity.architecture must be MINAKANUSHI, got {self.architecture}")
        if self.organization != "NULLXES":
            raise ValueError(f"identity.organization must be NULLXES, got {self.organization}")
        if self.native_runtime != "nullxes":
            raise ValueError(f"identity.native_runtime must be nullxes, got {self.native_runtime}")


@dataclass(frozen=True)
class NpfConfig:
    num_frequencies: int = 32
    max_sources: int = 64
    time_scale_seconds: float = 1.0


@dataclass(frozen=True)
class CognitionConfig:
    budget: int = 4
    convergence_threshold: float = 0.02


@dataclass(frozen=True)
class PersistenceConfig:
    steps: int = 8
    retirement_uncertainty: float = 0.95


@dataclass(frozen=True)
class HorizonConfig:
    immediate: int = 1
    short: int = 4
    medium: int = 8

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.immediate, self.short, self.medium)


@dataclass(frozen=True)
class ArchitectureConfig:
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    latent_dim: int = 2048
    state_dim: int = 2048
    memory_dim: int = 2048
    world_slots: int = 512
    memory_slots: int = 1024
    core_depth: int = 24
    uncertainty_channels: int = 8
    dropout: float = 0.0
    npf: NpfConfig = field(default_factory=NpfConfig)
    cognition: CognitionConfig = field(default_factory=CognitionConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    prediction_horizons: HorizonConfig = field(default_factory=HorizonConfig)
    dt: float = 0.1
    max_sources: int = 64
    max_observations: int = 64
    future_branches: int = 3

    def validate(self) -> None:
        self.identity.validate()
        positive = {
            "latent_dim": self.latent_dim,
            "state_dim": self.state_dim,
            "memory_dim": self.memory_dim,
            "world_slots": self.world_slots,
            "memory_slots": self.memory_slots,
            "core_depth": self.core_depth,
            "uncertainty_channels": self.uncertainty_channels,
            "max_sources": self.max_sources,
            "max_observations": self.max_observations,
            "future_branches": self.future_branches,
        }
        for name, value in positive.items():
            if int(value) < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if self.latent_dim != self.state_dim:
            raise ValueError("milestone-1 requires latent_dim == state_dim")
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.memory_dim != self.latent_dim:
            raise ValueError("milestone-1 requires memory_dim == latent_dim so memory writes stay in the same space")


@dataclass(frozen=True)
class LossLambdaConfig:
    state: float = 1.0
    temporal: float = 1.0
    future: float = 0.5
    uncertainty: float = 0.3
    causal: float = 0.2
    memory: float = 0.4
    action: float = 0.3
    representation: float = 0.05
    belief: float = 0.0
    revision: float = 0.0


@dataclass(frozen=True)
class RegularizerConfig:
    isotropic_weight: float = 0.05
    counterfactual_margin: float = 0.25


@dataclass(frozen=True)
class TrainingConfig:
    stage: int = 0
    name: str = "architecture_validation"
    architecture: str = "configs/architecture/cpu_dev.yaml"
    simulation: str = "configs/simulation/milestone1.yaml"
    seed: int = 7
    steps: int = 200
    batch_size: int = 8
    sequence_length: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    log_every: int = 20
    eval_every: int = 50
    checkpoint_every: int = 100
    precision: str = "float32"
    device: str = "cpu"
    parallelism: str = "none"
    activation_checkpoint: bool = False
    warmup_steps: int = 0
    shard_max_bytes: int = 0
    lambdas: LossLambdaConfig = field(default_factory=LossLambdaConfig)
    regularizer: RegularizerConfig = field(default_factory=RegularizerConfig)
    n_overfit_episodes: int = 16
    dataset_name: str = "stage0_synthetic"
    dataset_root: str = ""


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cpu"
    seed: int = 7
    deterministic: bool = True
    log_level: str = "INFO"
    nan_guard: bool = True
    latency_telemetry: bool = True


@dataclass(frozen=True)
class RestrictedZone:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class SimulationConfig:
    name: str = "milestone1"
    dt: float = 0.1
    arena: tuple[float, float, float, float] = (0.0, 10.0, 0.0, 10.0)
    agent_start: tuple[float, float] = (1.0, 1.0)
    home: tuple[float, float] = (1.0, 1.0)
    sensor_range: float = 3.5
    sensor_noise_std: float = 0.08
    occlusion: bool = True
    max_speed: float = 1.2
    n_movers: int = 3
    n_obstacles: int = 2
    n_targets: int = 2
    restricted_zones: tuple[RestrictedZone, ...] = (
        RestrictedZone(7.0, 7.0, 9.5, 9.5),
    )
    movers: tuple[dict[str, Any], ...] = ()
    obstacles: tuple[dict[str, Any], ...] = ()
    targets: tuple[dict[str, Any], ...] = ()
    hard_constraints: tuple[str, ...] = (
        "stay_in_arena",
        "no_enter_restricted_zone",
        "no_collide_obstacle",
        "max_speed",
    )


@dataclass(frozen=True)
class MinakanushiConfig:
    architecture: ArchitectureConfig
    training: TrainingConfig | None = None
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    def validate(self) -> None:
        self.architecture.validate()


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a mapping")
    return data


def _identity_from(raw: dict[str, Any]) -> IdentityConfig:
    block = raw.get("identity", {})
    return IdentityConfig(**block) if block else IdentityConfig()


def load_architecture(path: str | Path) -> ArchitectureConfig:
    raw = _read_yaml(Path(path))
    npf_raw = raw.get("npf", {})
    cog_raw = raw.get("cognition", {})
    per_raw = raw.get("persistence", {})
    hor_raw = raw.get("prediction_horizons", {})
    return ArchitectureConfig(
        identity=_identity_from(raw),
        latent_dim=int(raw["latent_dim"]),
        state_dim=int(raw["state_dim"]),
        memory_dim=int(raw["memory_dim"]),
        world_slots=int(raw["world_slots"]),
        memory_slots=int(raw["memory_slots"]),
        core_depth=int(raw["core_depth"]),
        uncertainty_channels=int(raw["uncertainty_channels"]),
        dropout=float(raw.get("dropout", 0.0)),
        npf=NpfConfig(**npf_raw) if npf_raw else NpfConfig(),
        cognition=CognitionConfig(**cog_raw) if cog_raw else CognitionConfig(),
        persistence=PersistenceConfig(**per_raw) if per_raw else PersistenceConfig(),
        prediction_horizons=HorizonConfig(**hor_raw) if hor_raw else HorizonConfig(),
        dt=float(raw.get("dt", 0.1)),
        max_sources=int(raw.get("max_sources", 64)),
        max_observations=int(raw.get("max_observations", 64)),
        future_branches=int(raw.get("future_branches", 3)),
    )


def load_training(path: str | Path) -> TrainingConfig:
    raw = _read_yaml(Path(path))
    lambdas = raw.get("lambdas", {})
    regularizer = raw.get("regularizer", {})
    return TrainingConfig(
        stage=int(raw["stage"]),
        name=str(raw["name"]),
        architecture=str(raw["architecture"]),
        simulation=str(raw["simulation"]),
        seed=int(raw.get("seed", 7)),
        steps=int(raw["steps"]),
        batch_size=int(raw["batch_size"]),
        sequence_length=int(raw["sequence_length"]),
        learning_rate=float(raw["learning_rate"]),
        weight_decay=float(raw.get("weight_decay", 0.0)),
        grad_clip=float(raw.get("grad_clip", 1.0)),
        log_every=int(raw.get("log_every", 20)),
        eval_every=int(raw.get("eval_every", 50)),
        checkpoint_every=int(raw.get("checkpoint_every", 100)),
        precision=str(raw.get("precision", "float32")),
        device=str(raw.get("device", "cpu")),
        parallelism=str(raw.get("parallelism", "none")),
        activation_checkpoint=bool(raw.get("activation_checkpoint", False)),
        warmup_steps=int(raw.get("warmup_steps", 0)),
        shard_max_bytes=int(raw.get("shard_max_bytes", 0)),
        lambdas=LossLambdaConfig(**lambdas) if lambdas else LossLambdaConfig(),
        regularizer=RegularizerConfig(**regularizer) if regularizer else RegularizerConfig(),
        n_overfit_episodes=int(raw.get("n_overfit_episodes", 16)),
        dataset_name=str(raw.get("dataset_name", "stage0_synthetic")),
        dataset_root=str(raw.get("dataset_root", "")),
    )


def load_runtime(path: str | Path) -> RuntimeConfig:
    raw = _read_yaml(Path(path))
    return RuntimeConfig(**raw)


def load_simulation(path: str | Path) -> SimulationConfig:
    raw = _read_yaml(Path(path))
    zones = tuple(
        RestrictedZone(float(z[0]), float(z[1]), float(z[2]), float(z[3]))
        for z in raw.get("restricted_zones", [])
    )
    arena = tuple(float(v) for v in raw["arena"])
    return SimulationConfig(
        name=str(raw["name"]),
        dt=float(raw["dt"]),
        arena=(arena[0], arena[1], arena[2], arena[3]),
        agent_start=(float(raw["agent_start"][0]), float(raw["agent_start"][1])),
        home=(float(raw["home"][0]), float(raw["home"][1])),
        sensor_range=float(raw["sensor_range"]),
        sensor_noise_std=float(raw["sensor_noise_std"]),
        occlusion=bool(raw.get("occlusion", True)),
        max_speed=float(raw["max_speed"]),
        n_movers=int(raw["n_movers"]),
        n_obstacles=int(raw["n_obstacles"]),
        n_targets=int(raw["n_targets"]),
        restricted_zones=zones,
        movers=tuple(raw.get("movers", ())),
        obstacles=tuple(raw.get("obstacles", ())),
        targets=tuple(raw.get("targets", ())),
        hard_constraints=tuple(raw.get("hard_constraints", ())),
    )


def load_config(
    architecture_path: str | Path,
    *,
    training_path: str | Path | None = None,
    runtime_path: str | Path | None = None,
    simulation_path: str | Path | None = None,
) -> MinakanushiConfig:
    architecture = load_architecture(architecture_path)
    training = load_training(training_path) if training_path is not None else None
    runtime = load_runtime(runtime_path) if runtime_path is not None else RuntimeConfig()
    if simulation_path is not None:
        simulation = load_simulation(simulation_path)
    elif training is not None:
        simulation = load_simulation(training.simulation)
    else:
        simulation = SimulationConfig()
    config = MinakanushiConfig(
        architecture=architecture,
        training=training,
        runtime=runtime,
        simulation=simulation,
    )
    config.validate()
    return config
