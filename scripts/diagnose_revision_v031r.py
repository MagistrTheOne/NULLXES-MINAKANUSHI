"""v0.3.1-R revision-trigger diagnostic. No train. No 6.8B on CPU.

CPU (always):

    python scripts/diagnose_revision_v031r.py
    python scripts/diagnose_revision_v031r.py --dataset dataset/mina_6_8b_v03
    python scripts/diagnose_revision_v031r.py \\
      --before-verdict artifacts/v031/verdict/step128.json \\
      --after-verdict artifacts/v031/verdict/step1128.json

H200 live, sensor_delay heldout only (not the full 100):

    python scripts/diagnose_revision_v031r.py \\
      --dataset dataset/mina_6_8b_v03 \\
      --before /workspace/checkpoints/minakanushi_stage0_step128.mina \\
      --after experiments/mina_6_8b_v031/minakanushi_stage0_step1128.mina \\
      --out artifacts/v031r/live.json
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path

from minakanushi.training.revision_forensic import diagnose, live_slot_audit
from minakanushi.training.v031_verdict import write_report

ROOT = Path(__file__).resolve().parents[1]


def _live_one(mina: Path, *, config: Path, dataset: Path) -> dict:
    import torch

    from minakanushi.architecture.config import load_config, load_training
    from minakanushi.architecture.freeze import is_6_8b_profile
    from minakanushi.training.checkpoint import load_mina
    from minakanushi.training.episode_dataset import JsonEpisodeDataset
    from minakanushi.training.trainer import Trainer
    from minakanushi.training.v031_verdict import _clear_stale_torchrun, refuse_cpu_6_8b

    _clear_stale_torchrun()
    training = load_training(config)
    cfg = load_config(
        ROOT / training.architecture,
        training_path=config,
        runtime_path=ROOT / "configs" / "runtime" / "cpu.yaml",
        simulation_path=ROOT / training.simulation,
    )
    train = replace(cfg.training, parallelism="none", dataset_split="heldout", activation_checkpoint=False)
    trainer = Trainer(replace(cfg, training=train), ROOT, eval_only=True)
    refuse_cpu_6_8b(trainer)
    if is_6_8b_profile(trainer.config.architecture) and trainer.device.type != "cuda":
        raise RuntimeError("v0.3.1-R live dump is H200/B300 only")
    print(f"construct ok device={trainer.device} loading {mina}", flush=True)
    load_mina(Path(mina), trainer.system, optimizer=None)
    trainer.system.eval()
    held = JsonEpisodeDataset(dataset, seed=11, split="heldout")
    rows = []
    for i, name in enumerate(held.scenarios):
        if name != "sensor_delay":
            continue
        episode = held.episode(i)
        with torch.no_grad():
            pkt = trainer.unroll(i + 1, episode=episode)
        row = live_slot_audit(pkt)
        row["index"] = i
        rows.append(row)
        print(
            f"sensor_delay {len(rows)} {mina.name} cut={row['cut']} "
            f"n_mover={row['n_mover_evidence']} n_need={row['n_need']} "
            f"max_before_d={row['max_before_d']:.4f} moved={row['mean_moved']:.4f} "
            f"detected={row['revision_detected']:.2f}",
            flush=True,
        )
    n = len(rows) or 1
    suppressed = sum(1 for r in rows if r["cut"] == "teacher_suppressed")
    no_mover = sum(1 for r in rows if r["cut"] == "no_mover_evidence")
    summary = {
        "checkpoint": str(mina),
        "n": len(rows),
        "no_mover_evidence_rate": no_mover / n if rows else float("nan"),
        "teacher_suppressed_rate": suppressed / n if rows else float("nan"),
        "mean_max_before_d": sum(r["max_before_d"] for r in rows) / n if rows else float("nan"),
        "mean_n_mover_evidence": sum(r["n_mover_evidence"] for r in rows) / n if rows else float("nan"),
        "mean_n_need": sum(r["n_need"] for r in rows) / n if rows else float("nan"),
        "mean_detected": sum(r["revision_detected"] for r in rows) / n if rows else float("nan"),
        "mean_constructor_corrections": sum(r["n_constructor_corrections"] for r in rows) / n if rows else float("nan"),
        "rows": rows,
    }
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--before-verdict", type=Path, default=None)
    parser.add_argument("--after-verdict", type=Path, default=None)
    parser.add_argument("--before", type=Path, default=None)
    parser.add_argument("--after", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "training" / "mina_6_8b_v03.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "v031r" / "diagnostic.json")
    args = parser.parse_args()

    report = diagnose(
        dataset=args.dataset,
        before_verdict=args.before_verdict,
        after_verdict=args.after_verdict,
    )
    if args.before is not None or args.after is not None:
        if args.before is None or args.after is None or args.dataset is None:
            raise SystemExit("live dump needs --before, --after, and --dataset")
        if not args.before.is_file() or not args.after.is_file():
            raise SystemExit("both --before and --after *.mina must exist on this machine")
        print(f"BEFORE {args.before}", flush=True)
        report["live_before"] = _live_one(args.before, config=args.config, dataset=args.dataset)
        print(f"AFTER {args.after}", flush=True)
        report["live_after"] = _live_one(args.after, config=args.config, dataset=args.dataset)
        b = report["live_before"]["mean_max_before_d"]
        a = report["live_after"]["mean_max_before_d"]
        report["live_compare"] = {
            "max_before_d_dropped": bool(a < b),
            "after_no_mover": report["live_after"]["no_mover_evidence_rate"] >= 0.8,
            "after_teacher_suppressed": report["live_after"]["teacher_suppressed_rate"] >= 0.8,
            "hypothesis_holds": bool(
                report["live_after"]["no_mover_evidence_rate"] >= 0.8
                or (a < b and report["live_after"]["teacher_suppressed_rate"] >= 0.8)
            ),
        }
    write_report(args.out, report)
    print(json.dumps({k: report[k] for k in report if k not in {"heldout_geometry", "generated"}}, indent=2, sort_keys=True))
    cpu = report["cpu_verdict"]
    print(
        f"v0.3.1-R CPU: no_mover={cpu['train_frame_has_no_mover']} "
        f"timestamp_only={cpu['delay_is_timestamp_only']} "
        f"oracle_prev_teacher_dead={cpu['oracle_prev_teacher_dead']}"
    )
    if report.get("live_compare"):
        print(f"v0.3.1-R live hypothesis_holds={report['live_compare']['hypothesis_holds']}")


if __name__ == "__main__":
    main()
