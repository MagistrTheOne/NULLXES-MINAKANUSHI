"""Walk 100 v0.3 episodes through the warm/intelligence sampler. CPU. No 6.8B.

    python scripts/gate_v031_validate.py --root dataset/mina_6_8b_v03 --n 100
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from minakanushi.training.episode_dataset import JsonEpisodeDataset
from minakanushi.training.phase_sampler import PhaseCurriculumSampler, mode_for_job_step
from simulations.synthetic_world.dataset_v1 import validate_episode_record

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "dataset" / "mina_6_8b_v03")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--warm-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    ds = JsonEpisodeDataset(args.root, seed=args.seed)
    sampler = PhaseCurriculumSampler(ds.paths, ds.phases, seed=args.seed)
    counts: Counter[str] = Counter()
    pwm = False
    diversity = 0
    for step in range(1, int(args.n) + 1):
        mode = mode_for_job_step(step, warm_steps=args.warm_steps)
        idx = sampler.choose(step, mode)
        rec = ds.record(idx)
        validate_episode_record(rec, curriculum_6_8b=True)
        counts[str(rec.get("phase"))] += 1
        pwm = pwm or bool((rec.get("embodiment") or {}).get("pwm"))
        if float(rec.get("future_diversity") or 0.0) > 1e-6:
            diversity += 1
    report = {
        "n": int(args.n),
        "phase_counts": dict(counts),
        "pwm": pwm,
        "future_diversity_positive": diversity,
        "warm_steps": int(args.warm_steps),
        "pass": (not pwm) and diversity == int(args.n) and set(counts) == {"physics", "agency", "causality", "embodiment"},
    }
    if not report["pass"]:
        raise SystemExit(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
