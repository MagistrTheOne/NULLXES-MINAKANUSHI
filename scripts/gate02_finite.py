"""Gate 02 finite forward / backward / live-loss probe. cpu_dev + stage0_overfit."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from minakanushi.training.trainer import trainer_from_files


ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    "NPF": "position_field",
    "DWC": "world_core",
    "MemoryEngine": "memory",
    "UncertaintyEngine": "uncertainty",
    "FutureEngine": "future",
}
NON_LEARNED = ("ConstraintKernel", "ActionPolicy", "StrategyEngine")
LOSS_TARGETS = {
    "state": "DWC",
    "temporal": "DWC",
    "future": "FutureEngine",
    "uncertainty": "UncertaintyEngine",
    "causal": "DWC",
    "memory": "MemoryEngine",
    "action": "FutureEngine",
    "representation": "DWC",
}


def _report_tensor(name: str, t: torch.Tensor) -> dict:
    finite = bool(torch.isfinite(t).all().item()) if t.numel() else True
    return {
        "name": name,
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "finite": finite,
    }


def _module_grad(module: torch.nn.Module) -> dict:
    norms = []
    has = False
    finite = True
    for p in module.parameters():
        if p.grad is None:
            continue
        has = True
        if not torch.isfinite(p.grad).all():
            finite = False
        norms.append(float(p.grad.detach().norm().item()))
    return {
        "has_grad": has,
        "gradient_norm": float(sum(norms) ** 0.5) if norms else 0.0,
        "finite_grad": finite if has else True,
    }


def main() -> None:
    trainer = trainer_from_files(ROOT, ROOT / "configs" / "training" / "stage0_overfit.yaml")
    system = trainer.system
    system.train()
    pkt = trainer.unroll(1)
    reports = [
        _report_tensor("NPF.embedding", pkt.pos.embedding),
        _report_tensor("NPF.temporal", pkt.pos.temporal_embedding),
        _report_tensor("NPF.spatial", pkt.pos.spatial_embedding),
        _report_tensor("DWC.latent_state", pkt.pred.latent_state),
        _report_tensor("memory.read", pkt.hints),
        _report_tensor("memory.write", pkt.writes),
        _report_tensor("uncertainty", pkt.pred.uncertainty),
        _report_tensor("future.trajectory", pkt.pred_future),
        _report_tensor("composite.total", pkt.breakdown.total),
    ]
    logits = [t.branch_logit for t in pkt.trajs if t.branch_logit is not None]
    if logits:
        reports.append(_report_tensor("future.branch_logits", torch.stack(logits)))
    scores = torch.tensor([float(t.probability.detach()) for t in pkt.trajs])
    reports.append(_report_tensor("strategy.branch_probabilities", scores))
    for name, term in pkt.breakdown.terms.items():
        reports.append(_report_tensor(f"loss.{name}", term))

    bad = [r for r in reports if not r["finite"]]
    print("FORWARD " + json.dumps(reports, ensure_ascii=True))
    if bad:
        raise RuntimeError(f"NaN/Inf in forward: {bad}")

    # Snapshot params, one optimizer step, detect updates.
    before = {n: p.detach().clone() for n, p in system.named_parameters()}
    trainer.opt.zero_grad(set_to_none=True)
    pkt.breakdown.total.backward()
    grads = {}
    for label, attr in MODULES.items():
        grads[label] = _module_grad(getattr(system, attr))
    trainer.opt.step()
    for label, attr in MODULES.items():
        mod = getattr(system, attr)
        changed = False
        for n, p in mod.named_parameters():
            key = None
            for full, tensor in before.items():
                if tensor.data_ptr() != p.data_ptr() and full.endswith(n) and getattr(system, attr) is mod:
                    pass
            for full, tensor in before.items():
                if p.shape == tensor.shape and torch.equal(p.detach(), tensor):
                    continue
            # Compare by object identity via named_parameters of the submodule.
        for n, p in mod.named_parameters():
            full = f"{attr}.{n}"
            if full in before and not torch.equal(p.detach(), before[full]):
                changed = True
                break
            # named_parameters on submodule doesn't include prefix
            matches = [k for k in before if k.endswith(n) and before[k].shape == p.shape]
            if matches and not torch.equal(p.detach(), before[matches[0]]):
                changed = True
                break
        grads[label]["parameter_update_detected"] = changed
        print(f"GRAD {label} {json.dumps(grads[label])}")

    print("NON_LEARNED " + json.dumps(list(NON_LEARNED)))

    # Per-loss LIVE check: new unroll, backward each term.
    live = {}
    pkt2 = trainer.unroll(1)
    for name, target in LOSS_TARGETS.items():
        trainer.opt.zero_grad(set_to_none=True)
        term = pkt2.breakdown.terms[name]
        if not torch.isfinite(term).all():
            live[name] = {"value": None, "finite": False, "target_module": target, "nonzero_gradient": False}
            continue
        term.backward(retain_graph=True)
        attr = MODULES[target]
        info = _module_grad(getattr(system, attr))
        live[name] = {
            "value": float(term.detach()),
            "finite": True,
            "target_module": target,
            "nonzero_gradient": bool(info["has_grad"] and info["gradient_norm"] > 0.0),
            "gradient_norm": info["gradient_norm"],
        }
        print(f"LOSS {name} {json.dumps(live[name])}")

    (ROOT / "experiments" / "stage0_overfit").mkdir(parents=True, exist_ok=True)
    out = ROOT / "experiments" / "stage0_overfit" / "gate02_finite.json"
    out.write_text(json.dumps({"forward": reports, "grads": grads, "live": live}, indent=2), encoding="utf-8")
    if any(not x["finite"] for x in live.values()):
        raise RuntimeError("non-finite loss term")
    missing = [k for k, v in live.items() if not v["nonzero_gradient"] and k not in {"action"}]
    print("LIVE_SUMMARY " + json.dumps({k: v["nonzero_gradient"] for k, v in live.items()}))
    if missing:
        print("WARN live losses without target-module grad: " + ",".join(missing))


if __name__ == "__main__":
    main()
