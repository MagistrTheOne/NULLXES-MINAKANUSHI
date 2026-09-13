#!/usr/bin/env python3
"""YMBOT-C laboratory runner.

Tracks:
  L0  closed loop + selftest   (this folder only, CPU)
  L1  tiny world residual      (this folder, CPU or lab GPU — not 6.8B)
  L2  6.8B infer               (partner wheel + 80GB-class GPU + Hub checkpoint)

Does not train minakanushi_6_8b. Does not construct 6.8B on CPU / 4090 / 6000.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mina_loop import IDENTITY, dumps, run_closed_loop, run_selftest, train_world_residual

HUB_MODEL = "https://huggingface.co/MagistrTheOne/MINAKANUSHI-6.8B"
HUB_DATA = "https://huggingface.co/datasets/MagistrTheOne/mina-6.8b-v03"


def _write(path: Path | None, payload: dict) -> None:
    text = dumps(payload)
    print(text)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


def cmd_l0(args: argparse.Namespace) -> int:
    if args.selftest:
        report = run_selftest()
        _write(args.report, report)
        return 0 if report["passed"] else 1
    report = run_closed_loop(steps=args.steps, seed=args.seed, policy_enabled=not args.policy_off)
    _write(args.report, report)
    return 0


def cmd_l1(args: argparse.Namespace) -> int:
    try:
        report = train_world_residual(steps=args.steps, device=args.device, seed=args.seed)
    except RuntimeError as exc:
        print(json.dumps({"ok": False, "track": "L1", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    _write(args.report, report)
    return 0


def cmd_l2(args: argparse.Namespace) -> int:
    """Real 6.8B infer path. Refuses rather than constructing on a laptop."""
    try:
        import torch
    except ImportError:
        print("L2 requires PyTorch.", file=sys.stderr)
        return 2

    if args.checkpoint is None:
        print(
            dumps(
                {
                    "ok": False,
                    "track": "L2",
                    "error": "missing --checkpoint",
                    "hub_model": HUB_MODEL,
                    "hub_data": HUB_DATA,
                    "canonical": "*.mina  (safetensors is a weight mirror only)",
                    "refuse": "this kit must not construct minakanushi_6_8b on CPU/4090/6000",
                }
            ),
            file=sys.stderr,
        )
        return 2

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        print(f"checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2

    try:
        from minakanushi.architecture.config import load_architecture
        from minakanushi.architecture.model import MinakanushiSystem
        from minakanushi.training.checkpoint import load_mina
    except ImportError:
        print(
            "L2 requires the NULLXES minakanushi partner wheel, not YMBOT-C L0 alone.\n"
            "Install the wheel NULLXES shipped with this kit, then retry.",
            file=sys.stderr,
        )
        return 2

    if not torch.cuda.is_available():
        print("L2 refused: CUDA is required to load 6.8B. CPU construct is forbidden.", file=sys.stderr)
        return 2
    vram_gb = float(torch.cuda.get_device_properties(0).total_memory) / (1024**3)
    if vram_gb < 70.0:
        print(
            f"L2 refused: {vram_gb:.1f} GB VRAM. Need ≥80 GB class (A100-80 / H100 / H800 / 6000-96 / H20) for infer. "
            "Do not train 6.8B here.",
            file=sys.stderr,
        )
        return 2

    # Infer only. Architecture YAML must be the frozen 6.8B profile from the partner wheel.
    root = Path(args.arch) if args.arch else None
    if root is None:
        print("L2 refused: pass --arch path to configs/architecture/minakanushi_6_8b.yaml from the partner wheel.", file=sys.stderr)
        return 2
    config = load_architecture(root)
    if int(config.latent_dim) != 4096 or int(config.core_depth) != 32:
        print(
            f"L2 refused: unexpected profile latent_dim={config.latent_dim} depth={config.core_depth}. "
            "YMBOT-C L2 loads minakanushi_6_8b only.",
            file=sys.stderr,
        )
        return 2

    device = torch.device("cuda")
    system = MinakanushiSystem(config).to(device)
    load_mina(checkpoint, system)
    n_param = sum(p.numel() for p in system.parameters())
    report = {
        "ok": True,
        "track": "L2",
        "identity": {**IDENTITY, "lab_track": "L2"},
        "checkpoint": str(checkpoint),
        "params": int(n_param),
        "vram_gb": vram_gb,
        "device": str(torch.cuda.get_device_name(0)),
        "train": False,
        "accepted": False,
        "note": "Weights loaded. Official A/B/C verdict is the heldout-100 ledger, not this smoke load.",
    }
    _write(args.report, report)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="YMBOT-C MINAKANUSHI laboratory runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("l0", help="closed loop / selftest (CPU)")
    p0.add_argument("--steps", type=int, default=24)
    p0.add_argument("--seed", type=int, default=11)
    p0.add_argument("--policy-off", action="store_true")
    p0.add_argument("--selftest", action="store_true")
    p0.add_argument("--report", type=Path, default=None)
    p0.set_defaults(func=cmd_l0)

    p1 = sub.add_parser("l1", help="tiny residual train (not 6.8B)")
    p1.add_argument("--steps", type=int, default=40)
    p1.add_argument("--seed", type=int, default=11)
    p1.add_argument("--device", default="cpu")
    p1.add_argument("--report", type=Path, default=None)
    p1.set_defaults(func=cmd_l1)

    p2 = sub.add_parser("l2", help="6.8B infer only, 80GB-class GPU")
    p2.add_argument("--checkpoint", type=Path, default=None)
    p2.add_argument("--arch", type=Path, default=None)
    p2.add_argument("--report", type=Path, default=None)
    p2.set_defaults(func=cmd_l2)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
