"""HF safetensors export + reload + AutoConfig type tag. Not CausalLM.

    python scripts/gate_v031_export.py --mina probe.mina --out artifacts/v031/hf_probe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minakanushi.hub import HUB_MODEL_TYPE, refuse_if_research_scale
from minakanushi.training.export_roundtrip import export_and_reload

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mina", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/v031/hf_probe"))
    parser.add_argument("--cards-only", action="store_true")
    parser.add_argument("--shard-bytes", type=int, default=64)
    args = parser.parse_args()
    result = export_and_reload(
        args.mina,
        args.out,
        shard_bytes=args.shard_bytes,
        cards_only=args.cards_only,
    )
    from minakanushi.hub import register_minakanushi

    config_cls, model_cls = register_minakanushi()
    cfg = config_cls(model_type=HUB_MODEL_TYPE, latent_dim=4096)
    try:
        refuse_if_research_scale(cfg)
        raise SystemExit("6.8B construct must be refused")
    except RuntimeError as exc:
        if "load_mina" not in str(exc):
            raise
    report = {
        **result,
        "auto_config": config_cls.__name__,
        "auto_model": model_cls.__name__,
        "causal_lm": False,
        "runtime": "load_mina",
        "license": str((args.out / "LICENSE").exists()),
        "minakanushi_config": str((args.out / "minakanushi_config.json").exists()),
    }
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
