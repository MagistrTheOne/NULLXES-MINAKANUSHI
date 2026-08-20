"""PersonaModel — external presentation only.

Does not enter WorldState, FutureEngine, StrategyEngine, ConstraintKernel,
or ActionPolicy. Changing persona must not change belief or intent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from minakanushi.identity.constants import ARCHITECTURE_NAME, ORGANIZATION, SHORT_NAME


@dataclass
class PersonaModel:
    short_name: str = SHORT_NAME
    full_name: str = ARCHITECTURE_NAME
    organization: str = ORGANIZATION
    feminine_presenting: bool = True
    visual_profile: str = "unspecified"
    voice_profile: str = "unspecified"
    communication_style: str = "precise_operational"
    explanation_style: str = "causal_then_action"
    reporting_style: str = "telemetry_first"
    communication_preferences: tuple[str, ...] = ("no_roleplay", "no_human_imitation")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> PersonaModel:
        data = dict(raw)
        prefs = data.get("communication_preferences")
        if isinstance(prefs, list):
            data["communication_preferences"] = tuple(prefs)
        known = {k for k in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def affects_cognition(self) -> bool:
        return False
