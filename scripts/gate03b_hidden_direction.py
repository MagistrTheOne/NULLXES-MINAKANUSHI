"""Gate 03B — hidden_correction direction diagnostic.

Not a new gate. Not an architecture change. Not a λ bump.

Question: is hidden direction 0.5 a cpu_dev/single-seed artifact, or a
stable physics-prior vs evidence conflict on gpu_train_v01?

Does not construct minakanushi_6_8b. Does not train unless --train.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

from minakanushi.training.checkpoint import load_mina
from minakanushi.training.trainer import trainer_from_files

ROOT = Path(__file__).resolve().parents[1]
CLASSES: tuple[str, ...] = ("hidden_correction", "conflict", "reacquisition")
DIRECTION_LIVE = 0.7


def _exam():
    path = Path(__file__).with_name("gate03_revision_validate.py")
    spec = importlib.util.spec_from_file_location("gate03_revision_validate", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0, "p10": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    ordered = sorted(values)
    return {
        "n": float(len(values)),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "std": float(std),
        "p10": float(ordered[max(0, int(0.10 * (len(ordered) - 1)))]),
        "p90": float(ordered[min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))]),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _term_means(rows: list[dict]) -> dict[str, float]:
    keys = ("revision", "state", "future", "belief", "temporal")
    out: dict[str, float] = {}
    for key in keys:
        vals = [float(r["terms"].get(key, 0.0)) for r in rows]
        out[key] = float(statistics.mean(vals)) if vals else 0.0
    rev = max(out["revision"], 1e-8)
    out["state_over_revision"] = out["state"] / rev
    out["future_over_revision"] = out["future"] / rev
    return out


def diagnose(trainer, n: int, seed0: int) -> dict:
    exam = _exam()
    trainer.system.eval()
    identity = trainer.config.architecture.identity
    if identity.architecture != "MINAKANUSHI":
        raise ValueError(f"identity {identity.architecture!r} is not MINAKANUSHI")
    lam = float(trainer.config.training.lambdas.revision)
    if abs(lam - 1.0) > 1e-9:
        raise ValueError(f"λ_revision={lam}, Gate 03B requires 1.0")
    by_class: dict[str, dict] = {}
    for name in CLASSES:
        rows = []
        for i in range(n):
            rows.append(exam.evaluate_scenario(trainer, name, episode_index=seed0 + i))
        direction = [r["revision_direction_accuracy"] for r in rows]
        detected = [r["revision_detected"] for r in rows]
        false_r = [r["false_revision_rate"] for r in rows]
        mag = [r["revision_magnitude_error"] for r in rows]
        latency = [r["revision_latency"] for r in rows]
        by_class[name] = {
            "direction": _mean_std(direction),
            "detected": _mean_std(detected),
            "false_revision": _mean_std(false_r),
            "magnitude_error": _mean_std(mag),
            "revision_latency": _mean_std(latency),
            "identity": _mean_std(
                [1.0 if r["identity"] == "same_hypothesis_revised" else 0.0 for r in rows]
            ),
            "terms": _term_means(rows),
            "direction_samples": direction,
        }
    hidden = by_class["hidden_correction"]["direction"]["mean"]
    if hidden >= DIRECTION_LIVE:
        verdict = "capacity_or_seed: hidden direction moved; Gate 03 closable after confirm"
    else:
        verdict = "stuck_prior: physics inertia vs evidence; next is term ratios, not λ"
    return {
        "gate": "03B_hidden_direction",
        "parent": "03_revision_validation",
        "architecture": identity.architecture,
        "organization": identity.organization,
        "latent_dim": trainer.config.architecture.latent_dim,
        "lambda_revision": lam,
        "n_per_class": n,
        "classes": by_class,
        "verdict": verdict,
        "hidden_direction_live": hidden >= DIRECTION_LIVE,
        "closed": ["minakanushi_6_8b", "H200", "architecture_change", "lambda_change", "new_slots"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 03B hidden_correction diagnostic")
    parser.add_argument(
        "--training",
        type=Path,
        default=ROOT / "configs" / "training" / "gate03_revision_validation.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--n", type=int, default=32, help="episodes per class. Blackwell: 1000")
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "gate03b")
    args = parser.parse_args()
    if args.n < 1:
        raise SystemExit("--n must be >= 1")
    trainer = trainer_from_files(ROOT, args.training)
    if args.checkpoint is not None:
        load_mina(args.checkpoint, trainer.system, optimizer=trainer.opt)
    report = diagnose(trainer, args.n, args.seed0)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gate03b_hidden_direction.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    hidden = report["classes"]["hidden_correction"]["direction"]
    print(json.dumps(report, indent=2))
    print()
    print(f"hidden direction mean={hidden['mean']:.3f} std={hidden['std']:.3f} n={int(hidden['n'])}")
    print(f"verdict: {report['verdict']}")


if __name__ == "__main__":
    main()
