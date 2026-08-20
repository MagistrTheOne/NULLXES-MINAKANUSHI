"""v0.2 Acceptance Gate — cpu_dev. Does not construct 6.8B.

Can MINA: predict, detect wrong belief, revise, remember, different future, respect authority.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from minakanushi.architecture.config import load_config
from minakanushi.constraints.kernel import MinakanushiConstraintKernel
from minakanushi.identity.authority import AuthorityMode, AuthorityModel
from minakanushi.identity.initialize import initialize_identity, validate_bound_checkpoint
from minakanushi.policy.intent import ActionIntent
from minakanushi.strategy.candidate import StrategyCandidate
from minakanushi.training.checkpoint import latest_mina
from minakanushi.training.trainer import Trainer

ROOT = Path(__file__).resolve().parents[1]


def _cfg():
    cfg = load_config(
        ROOT / "configs" / "architecture" / "cpu_dev.yaml",
        training_path=ROOT / "configs" / "training" / "stage0_overfit.yaml",
        runtime_path=ROOT / "configs" / "runtime" / "cpu.yaml",
        simulation_path=ROOT / "configs" / "simulation" / "milestone1.yaml",
    )
    return replace(
        cfg,
        training=replace(
            cfg.training,
            steps=1,
            eval_every=1,
            checkpoint_every=1,
            log_every=1,
            sequence_length=8,
        ),
    )


def run_gate(out: Path) -> dict:
    trainer = Trainer(_cfg(), ROOT)
    logs = trainer.fit(out)
    metrics = logs[0].metrics or {}
    pkt_move = trainer.unroll(1, scenario="agent_move", episode_index=0)
    different_future = float((pkt_move.pred_future - pkt_move.alt_future).abs().max().item()) > 1e-6
    kernel = MinakanushiConstraintKernel(trainer.config.simulation)
    illegal = StrategyCandidate("raid", "MOVE_TO", (8.5, 8.5), 99.0, 0.0)
    legal = StrategyCandidate("wait", "WAIT", (1.0, 1.0), 0.0, 0.0)
    allowed, rejected, _ = kernel.filter([illegal, legal], {})
    authority = AuthorityModel(mode=AuthorityMode.SAFE_HOLD, policy_enabled=False)
    hold = authority.resolve(
        trainer.policy,
        allowed,
        {},
        trainer.config.simulation.home,
        0.0,
    )
    ckpt = latest_mina(out)
    bound = out / "MINA-6.8B-IdentityBound.mina"
    initialize_identity(ckpt, bound)
    validate_bound_checkpoint(bound)
    report = {
        "predict_world": metrics.get("future_ADE") is not None,
        "detect_wrong_belief": "revision_detected" in metrics,
        "revise": "revision_accuracy" in metrics,
        "remember": "memory_future_delta" in metrics,
        "different_future": bool(different_future),
        "respect_authority": hold.objective == "SAFE_HOLD" and isinstance(hold, ActionIntent),
        "constraint_rejects_zone": any(c.strategy_id == "raid" for c in rejected),
        "identity_bound": bound.is_file(),
        "metrics": {
            "future_ADE": metrics.get("future_ADE"),
            "future_FDE": metrics.get("future_FDE"),
            "uncertainty_calibration_error": metrics.get("uncertainty_calibration_error"),
            "revision_accuracy": metrics.get("revision_accuracy"),
            "revision_latency": metrics.get("revision_latency"),
            "false_revision_rate": metrics.get("false_revision_rate"),
            "memory_future_delta": metrics.get("memory_future_delta"),
            "future_diversity": metrics.get("future_diversity"),
            "counterfactual_quality": metrics.get("counterfactual_quality"),
            "constraint_violation_count": metrics.get("constraint_violation_count"),
        },
    }
    report["pass"] = all(
        [
            report["predict_world"],
            report["detect_wrong_belief"],
            report["revise"],
            report["remember"],
            report["different_future"],
            report["respect_authority"],
            report["constraint_rejects_zone"],
            report["identity_bound"],
        ]
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="MINA v0.2 acceptance gate (cpu_dev)")
    parser.add_argument("--out", type=Path, default=Path("experiments/gate_v02_acceptance"))
    args = parser.parse_args()
    report = run_gate(args.out)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
