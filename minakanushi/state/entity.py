"""Entity helpers."""

from __future__ import annotations

from minakanushi.architecture.mina_unit import KIND_IDS

AGENT_SLOT = 0


def kind_name(kind_id: int) -> str:
    inverse = {v: k for k, v in KIND_IDS.items()}
    return inverse.get(kind_id, "unknown")
