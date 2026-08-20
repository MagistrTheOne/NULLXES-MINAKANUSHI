"""Hold-class strategies share zero velocity in the physical plant.

They do not share one action embedding. WAIT is not OBSERVE is not SAFE_HOLD.
"""

from __future__ import annotations

HOLD_MODE: dict[str, float] = {
    "WAIT": 0.0,
    "OBSERVE": 1.0,
    "SAFE_HOLD": 2.0,
    "ABORT": 3.0,
    "REQUEST_ASSISTANCE": 4.0,
}


def is_hold(objective: str) -> bool:
    return objective in HOLD_MODE


def hold_mode(objective: str) -> float:
    if objective not in HOLD_MODE:
        raise KeyError(f"not a hold objective: {objective}")
    return HOLD_MODE[objective]
