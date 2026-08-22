"""H200 capability verdict: step128 vs step1128 on sealed heldout.

Not more training. Does not construct 6.8B on CPU.

    python scripts/gate_v031_h200_verdict.py \\
      --before /workspace/checkpoints/minakanushi_stage0_step128.mina \\
      --after experiments/mina_6_8b_v031/minakanushi_stage0_step1128.mina \\
      --out artifacts/v031/verdict
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from minakanushi.training.episode_dataset import JsonEpisodeDataset
from minakanushi.training.v031_verdict import (
    compare_reports,
    eval_trainer,
    evaluate_heldout,
    write_report,
)

ROOT = Path(__file__).resolve().parents[1]


def _run_one(mina: Path, *, config: Path, dataset: Path, traces: int) -> dict:
    trainer = eval_trainer(ROOT, config, mina)
    held = JsonEpisodeDataset(dataset, seed=11, split="heldout")
    report = evaluate_heldout(trainer, held, traces=traces)
    report["checkpoint"] = str(mina)
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "training" / "mina_6_8b_v03.yaml")
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset" / "mina_6_8b_v03")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "v031" / "verdict")
    parser.add_argument("--traces", type=int, default=20)
    args = parser.parse_args()
    if not args.before.is_file() or not args.after.is_file():
        raise SystemExit("both --before and --after *.mina must exist on this machine")
    print(f"BEFORE {args.before}", flush=True)
    before = _run_one(args.before, config=args.config, dataset=args.dataset, traces=args.traces)
    write_report(args.out / "step128.json", before)
    print(f"AFTER {args.after}", flush=True)
    after = _run_one(args.after, config=args.config, dataset=args.dataset, traces=args.traces)
    write_report(args.out / "step1128.json", after)
    compare = compare_reports(before, after)
    write_report(args.out / "compare.json", compare)
    print(json.dumps(compare, indent=2, sort_keys=True), flush=True)
    if compare["variant"] == "A":
        print("VARIANT A — v0.3.1 accepted")
        return
    print(f"VARIANT {compare['variant']} — v0.3.1 not accepted")
    raise SystemExit(f"v0.3.1 verdict {compare['variant']}")


if __name__ == "__main__":
    main()
