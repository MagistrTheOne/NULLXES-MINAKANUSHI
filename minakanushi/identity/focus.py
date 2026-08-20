"""FocusState — what currently requires attention. Not a goal generator."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from minakanushi.state.entity import AGENT_SLOT
from minakanushi.state.world import WorldState


@dataclass
class FocusState:
    highest_uncertainty_area: str = "none"
    largest_prediction_error: str = "none"
    unfinished_process: str = "none"
    maintenance_need: str = "none"
    attention_target: str = "none"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> FocusState:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in known})


def focus_from_world(world: WorldState) -> FocusState:
    occ = world.occupied[0]
    if not bool(occ.any()):
        return FocusState(attention_target="empty_world")
    u = world.uncertainty[0].mean(dim=-1)
    mask = occ.clone()
    if occ.numel() > AGENT_SLOT:
        mask[AGENT_SLOT] = False
    if bool(mask.any()):
        slot = int(u.masked_fill(~mask, -1.0).argmax().item())
        eid = int(world.entity_id[0, slot].item())
        area = f"entity_{eid}"
        return FocusState(
            highest_uncertainty_area=area,
            attention_target=area,
            unfinished_process="none" if bool((world.age_unobserved[0, mask] == 0).all()) else "unobserved_tracks",
        )
    return FocusState(attention_target="embodiment")
