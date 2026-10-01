"""Single controller-speed contract for MINAKANUSHI motion.

`candidate.parameters["speed"]` is THE commanded speed everywhere:
StrategyEngine proposes it, FutureEngine predicts with it, SyntheticWorld
executes it, MaxSpeed constrains it. Default matches the legacy controller
(1.0); the platform cap is always `simulation.max_speed`.
"""

from __future__ import annotations

import numpy as np

DEFAULT_CONTROLLER_SPEED = 1.0


def commanded_speed(candidate) -> float:
    """Declared speed, defaulting to the legacy controller speed."""
    try:
        return float(candidate.parameters.get("speed", DEFAULT_CONTROLLER_SPEED))
    except (AttributeError, TypeError, ValueError):
        return DEFAULT_CONTROLLER_SPEED


def clamp_speed(speed: float, max_speed: float) -> float:
    return min(max(float(speed), 0.0), float(max_speed))


def command_velocity_np(delta: np.ndarray, speed: float, max_speed: float) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(delta))
    if norm < 1e-6:
        return np.zeros(2, dtype=np.float64)
    return (delta / norm) * clamp_speed(speed, max_speed)


def trajectory_peak_speed(agent_xy, dt: float) -> float:
    """Max per-step |dx|/dt over an agent trajectory. Pure python/sequences."""
    peak = 0.0
    for prev, nxt in zip(agent_xy[:-1], agent_xy[1:]):
        step = ((float(nxt[0]) - float(prev[0])) ** 2 + (float(nxt[1]) - float(prev[1])) ** 2) ** 0.5
        peak = max(peak, step / max(float(dt), 1e-9))
    return peak
