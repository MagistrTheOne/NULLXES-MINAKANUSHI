"""SelfModel — internal passport + operational state. Not a network."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field

from minakanushi.architecture.config import ArchitectureConfig, IdentityConfig
from minakanushi.identity.constants import (
    ARCHITECTURE_ID,
    ARCHITECTURE_NAME,
    NATIVE_RUNTIME,
    ORGANIZATION,
    SHORT_NAME,
    SYSTEM_CLASS,
)
from minakanushi.identity.experience import ExperienceLog, ExperienceRecord
from minakanushi.state.correction import CorrectionEvent


@dataclass
class IdentityBlock:
    architecture_name: str = ARCHITECTURE_NAME
    short_name: str = SHORT_NAME
    architecture_id: str = ARCHITECTURE_ID
    organization: str = ORGANIZATION
    system_class: str = SYSTEM_CLASS
    version: str = "0.1"
    native_runtime: str = NATIVE_RUNTIME


@dataclass
class InstanceBlock:
    instance_id: str
    creation_time: float
    runtime_age: float = 0.0
    history_reference: str = ""


@dataclass
class EmbodimentBlock:
    embodiment_id: str = "synthetic.platform"
    platform_type: str = "synthetic_agent"
    sensors: tuple[str, ...] = ("vector", "telemetry")
    actuators: tuple[str, ...] = ("intent_only",)
    capabilities: tuple[str, ...] = ("observe", "wait", "move_to", "safe_hold")
    limitations: tuple[str, ...] = ("no_raw_pwm", "no_language_cognition")


@dataclass
class ObjectiveBlock:
    active_objectives: tuple[str, ...] = ("maintain_world_belief",)
    mission_context: str = "milestone1_synthetic"


@dataclass
class ConstraintBlock:
    active_constraints: tuple[str, ...] = ()
    hard_limits: tuple[str, ...] = ()


@dataclass
class RuntimeBlock:
    health_state: str = "ok"
    resource_state: str = "ok"
    uncertainty_state: float = 0.5


@dataclass
class SelfModel:
    """This system. Never a WorldState entity slot."""

    identity: IdentityBlock = field(default_factory=IdentityBlock)
    instance: InstanceBlock = field(default_factory=lambda: InstanceBlock(str(uuid.uuid4()), time.time()))
    embodiment: EmbodimentBlock = field(default_factory=EmbodimentBlock)
    authority_mode: str = "AUTONOMOUS"
    policy_enabled: bool = True
    operator_connected: bool = False
    objectives: ObjectiveBlock = field(default_factory=ObjectiveBlock)
    constraints: ConstraintBlock = field(default_factory=ConstraintBlock)
    runtime: RuntimeBlock = field(default_factory=RuntimeBlock)
    experience: ExperienceLog = field(default_factory=ExperienceLog)

    def short_name(self) -> str:
        return self.identity.short_name

    def is_world_entity(self) -> bool:
        return False

    def tick(self, dt: float, uncertainty: float, corrections: tuple[CorrectionEvent, ...]) -> None:
        self.instance.runtime_age += float(dt)
        self.runtime.uncertainty_state = float(uncertainty)
        for event in corrections:
            self.experience.append(
                ExperienceRecord(
                    event_time=self.instance.creation_time + self.instance.runtime_age,
                    situation="belief_revision",
                    belief_before=str(event.old_xy),
                    action="revise",
                    result=str(event.new_xy),
                    belief_after=str(event.new_xy),
                    correction_required=True,
                )
            )

    def to_dict(self) -> dict:
        return {
            "identity": asdict(self.identity),
            "instance": asdict(self.instance),
            "embodiment": asdict(self.embodiment),
            "authority_mode": self.authority_mode,
            "policy_enabled": self.policy_enabled,
            "operator_connected": self.operator_connected,
            "objectives": asdict(self.objectives),
            "constraints": {
                "active_constraints": list(self.constraints.active_constraints),
                "hard_limits": list(self.constraints.hard_limits),
            },
            "runtime": asdict(self.runtime),
            "experience": self.experience.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> SelfModel:
        ident_raw = {k: v for k, v in dict(raw.get("identity", {})).items() if k in IdentityBlock.__dataclass_fields__}
        ident = IdentityBlock(**ident_raw)
        inst_raw = {k: v for k, v in dict(raw.get("instance", {})).items() if k in InstanceBlock.__dataclass_fields__}
        if "instance_id" not in inst_raw:
            inst_raw["instance_id"] = str(uuid.uuid4())
            inst_raw["creation_time"] = time.time()
        inst = InstanceBlock(**inst_raw)
        emb_raw = {k: v for k, v in dict(raw.get("embodiment", {})).items() if k in EmbodimentBlock.__dataclass_fields__}
        for key in ("sensors", "actuators", "capabilities", "limitations"):
            if key in emb_raw:
                emb_raw[key] = tuple(emb_raw[key])
        emb = EmbodimentBlock(**emb_raw) if emb_raw else EmbodimentBlock()
        obj_raw = dict(raw.get("objectives", {}))
        if "active_objectives" in obj_raw:
            obj_raw["active_objectives"] = tuple(obj_raw["active_objectives"])
        cons_raw = raw.get("constraints", {})
        model = cls(
            identity=ident,
            instance=inst,
            embodiment=emb,
            authority_mode=str(raw.get("authority_mode", "AUTONOMOUS")),
            policy_enabled=bool(raw.get("policy_enabled", True)),
            operator_connected=bool(raw.get("operator_connected", False)),
            objectives=ObjectiveBlock(**obj_raw) if obj_raw else ObjectiveBlock(),
            constraints=ConstraintBlock(
                active_constraints=tuple(cons_raw.get("active_constraints", ())),
                hard_limits=tuple(cons_raw.get("hard_limits", ())),
            ),
            runtime=RuntimeBlock(
                **{k: v for k, v in dict(raw.get("runtime", {})).items() if k in RuntimeBlock.__dataclass_fields__}
            ),
            experience=ExperienceLog.from_dict(raw.get("experience", {})),
        )
        return model

    @classmethod
    def from_config(cls, identity: IdentityConfig, architecture: ArchitectureConfig, hard_limits: tuple[str, ...] = ()) -> SelfModel:
        ident = IdentityBlock(
            architecture_name=identity.architecture,
            short_name=SHORT_NAME,
            architecture_id=ARCHITECTURE_ID,
            organization=identity.organization,
            system_class=identity.system_class,
            version=identity.architecture_version,
            native_runtime=identity.native_runtime,
        )
        return cls(
            identity=ident,
            constraints=ConstraintBlock(active_constraints=hard_limits, hard_limits=hard_limits),
        )
