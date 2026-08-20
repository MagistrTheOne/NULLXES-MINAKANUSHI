"""Gate 03 Revision Validation — cognitive exam, not a loss check.

MINA must abandon a wrong hypothesis when stronger evidence arrives.

Does not train unless --train is passed.
Does not construct minakanushi_6_8b.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from minakanushi.state.correction import CONFLICT_CHANNEL
from minakanushi.training.checkpoint import load_mina
from minakanushi.training.trainer import trainer_from_files
from simulations.synthetic_world.dataset import TRAIN_CURRICULUM

ROOT = Path(__file__).resolve().parents[1]
EXAM_SCENARIOS: tuple[str, ...] = ("hidden_correction", "conflict", "reacquisition")

PASS = {
    "revision_detected": 0.9,
    "false_revision_rate": 0.05,
    "revision_direction_accuracy": 0.7,
}


def _scenario_step(name: str) -> int:
    if name not in TRAIN_CURRICULUM:
        raise ValueError(f"{name} is not in TRAIN_CURRICULUM")
    return TRAIN_CURRICULUM.index(name) + 1


def _not_average(after: torch.Tensor, before: torch.Tensor, evidence: torch.Tensor, mask: torch.Tensor) -> float:
    """1 if after is closer to evidence than to the midpoint. Not (A+B)/2."""
    if not bool(mask.any()):
        return 0.0
    mid = 0.5 * (before + evidence)
    to_ev = torch.linalg.vector_norm(after - evidence, dim=-1)
    to_mid = torch.linalg.vector_norm(after - mid, dim=-1)
    hits = (to_ev < to_mid) & mask
    return float(hits.sum().item() / int(mask.sum().item()))


def evaluate_scenario(trainer, name: str) -> dict:
    step = _scenario_step(name)
    with torch.no_grad():
        pkt = trainer.unroll(step)
    metrics = trainer._metrics(pkt)
    pred = pkt.pred
    mask = pkt.should_revise
    corr = tuple(pred.corrections)
    corr_ids = {int(ev.entity_id) for ev in corr}
    occupied_ids = set()
    if bool(pred.occupied.any()):
        occupied_ids = {int(x) for x in pred.entity_id[0, pred.occupied[0]].tolist()}
    identity = "n/a"
    if name in ("hidden_correction", "reacquisition"):
        if pkt.n_constructor_corrections >= 1 and (not corr_ids or corr_ids <= occupied_ids):
            identity = "same_hypothesis_revised"
        else:
            identity = "new_object"
    conflict_u = 0.0
    if bool(mask.any()):
        conflict_u = float(pred.uncertainty[0, mask[0], CONFLICT_CHANNEL].mean().item())
    reasons = [ev.correction_reason for ev in corr]
    weights = [(float(ev.evidence_weight), float(ev.belief_weight)) for ev in corr]
    return {
        "scenario": name,
        "frame_index": pkt.frame_index,
        "constructor_corrections": pkt.n_constructor_corrections,
        "correction_reasons": reasons,
        "evidence_vs_belief_weights": weights,
        "identity": identity,
        "not_average": _not_average(pred.entity_xy, pkt.before_xy, pkt.evidence_xy, mask),
        "conflict_uncertainty": conflict_u,
        "revision_detected": metrics["revision_detected"],
        "revision_direction_accuracy": metrics["revision_direction_accuracy"],
        "revision_magnitude_error": metrics["revision_magnitude_error"],
        "revision_latency": metrics["revision_latency"],
        "false_revision_rate": metrics["false_revision_rate"],
        "belief_revision_accuracy": metrics["belief_revision_accuracy"],
        "should_revise": int(pkt.should_revise.sum().item()),
    }


def summarize(rows: list[dict]) -> dict:
    keys = (
        "revision_detected",
        "revision_direction_accuracy",
        "revision_magnitude_error",
        "revision_latency",
        "false_revision_rate",
    )
    mean = {k: sum(r[k] for r in rows) / max(len(rows), 1) for k in keys}
    detected_ok = all(r["revision_detected"] > PASS["revision_detected"] for r in rows)
    false_ok = all(r["false_revision_rate"] <= PASS["false_revision_rate"] for r in rows)
    direction_ok = all(
        r["revision_direction_accuracy"] > PASS["revision_direction_accuracy"] for r in rows
    )
    identity_ok = all(
        r["identity"] == "same_hypothesis_revised" for r in rows if r["scenario"] == "reacquisition"
    )
    conflict_ok = all(r["not_average"] > 0.0 for r in rows if r["scenario"] == "conflict")
    return {
        "mean": mean,
        "pass": {
            "revision_detected": detected_ok,
            "false_revision": false_ok,
            "direction_accuracy": direction_ok,
            "reacquisition_identity": identity_ok,
            "conflict_not_average": conflict_ok,
            "gate03": detected_ok and false_ok and direction_ok and identity_ok,
        },
    }


def run_exam(trainer) -> dict:
    trainer.system.eval()
    rows = [evaluate_scenario(trainer, name) for name in EXAM_SCENARIOS]
    report = {
        "gate": "03_revision_validation",
        "tag": "MINAKANUSHI-revision-gate",
        "lambda_revision": float(trainer.config.training.lambdas.revision),
        "architecture": trainer.config.architecture.identity.architecture,
        "latent_dim": trainer.config.architecture.latent_dim,
        "scenarios": rows,
        "summary": summarize(rows),
        "closed": ["minakanushi_6_8b", "H200", "humanoid", "yunmu", "large_datasets"],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 03 Revision Validation")
    parser.add_argument(
        "--training",
        type=Path,
        default=ROOT / "configs" / "training" / "gate03_revision_validation.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "gate03_revision")
    parser.add_argument("--train", action="store_true", help="short train after validation; off by default")
    args = parser.parse_args()
    trainer = trainer_from_files(ROOT, args.training)
    if args.checkpoint is not None:
        load_mina(args.checkpoint, trainer.system, optimizer=trainer.opt)
    report = run_exam(trainer)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gate03_revision.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    mean = report["summary"]["mean"]
    print(json.dumps(report, indent=2))
    print()
    print("| Metric | Value |")
    print("|---|---|")
    print(f"| revision_detected | {mean['revision_detected']:.3f} |")
    print(f"| direction_accuracy | {mean['revision_direction_accuracy']:.3f} |")
    print(f"| magnitude_error | {mean['revision_magnitude_error']:.3f} |")
    print(f"| latency | {mean['revision_latency']:.3f} |")
    print(f"| false_revision | {mean['false_revision_rate']:.3f} |")
    print(f"gate03 pass: {report['summary']['pass']['gate03']}")
    if args.train:
        if trainer.config.training.steps <= 0:
            raise SystemExit("--train requires training.steps > 0")
        trainer.system.train()
        trainer.fit(args.out)
        trainer.system.eval()
        after = run_exam(trainer)
        (args.out / "gate03_revision_after_train.json").write_text(
            json.dumps(after, indent=2), encoding="utf-8"
        )
        print("after train:", json.dumps(after["summary"], indent=2))


if __name__ == "__main__":
    main()
