"""Phase-weighted episode sampler. Data mix, not a new MINA layer.

Warm: physics is the world foundation.
Intelligence: revision / agency / embodiment after the job stabilizes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PHASE_ORDER: tuple[str, ...] = ("physics", "agency", "causality", "embodiment")

WARM_MIX: dict[str, float] = {
    "physics": 0.40,
    "causality": 0.30,
    "agency": 0.20,
    "embodiment": 0.10,
}

INTELLIGENCE_MIX: dict[str, float] = {
    "causality": 0.40,
    "agency": 0.30,
    "embodiment": 0.20,
    "physics": 0.10,
}


def mix_for_mode(mode: str) -> dict[str, float]:
    if mode == "warm":
        return dict(WARM_MIX)
    if mode == "intelligence":
        return dict(INTELLIGENCE_MIX)
    raise ValueError(f"unknown sampler mode {mode!r}")


def mode_for_job_step(job_step: int, *, warm_steps: int) -> str:
    if int(job_step) <= int(warm_steps):
        return "warm"
    return "intelligence"


class PhaseCurriculumSampler:
    def __init__(self, paths: tuple[Path, ...], phases: tuple[str, ...], *, seed: int) -> None:
        if len(paths) != len(phases):
            raise ValueError("paths / phases length mismatch")
        self.seed = int(seed)
        buckets: dict[str, list[int]] = {name: [] for name in PHASE_ORDER}
        for i, phase in enumerate(phases):
            key = str(phase)
            if key not in buckets:
                raise ValueError(f"unknown curriculum phase {key!r}")
            buckets[key].append(i)
        missing = [name for name, rows in buckets.items() if not rows]
        if missing:
            raise ValueError(f"sampler needs episodes in every phase, missing {missing}")
        self.indices: dict[str, tuple[int, ...]] = {name: tuple(rows) for name, rows in buckets.items()}

    def choose(self, step: int, mode: str) -> int:
        mix = mix_for_mode(mode)
        names = tuple(mix.keys())
        probs = np.asarray([mix[name] for name in names], dtype=np.float64)
        rng = np.random.default_rng(self.seed + 1_000_003 * int(step))
        phase = str(rng.choice(names, p=probs / probs.sum()))
        pool = self.indices[phase]
        return int(pool[int(rng.integers(0, len(pool)))])
