"""Reload a safetensors mirror. Does not construct 6.8B. Not CausalLM.

    python scripts/test_hf_reload.py --path MINAKANUSHI-6.8B
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from minakanushi.architecture.freeze import FROZEN_PARAM_ESTIMATE
from minakanushi.hub import HUB_MODEL_TYPE, refuse_if_research_scale, register_minakanushi
from minakanushi.training.export_roundtrip import _load_shards
from minakanushi.training.hf_export import assert_not_llm_card

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("MINAKANUSHI-6.8B"))
    args = parser.parse_args()
    root = args.path
    config_path = root / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"missing {config_path}. export on H200/B300 first.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert_not_llm_card(config)
    if config.get("model_type") != HUB_MODEL_TYPE:
        raise SystemExit(f"model_type={config.get('model_type')!r} is not minakanushi")
    parameters = int(config.get("parameters") or 0)
    shards = _load_shards(root)
    n_params = sum(int(t.numel()) for t in shards.values())
    config_cls, model_cls = register_minakanushi()
    cfg = config_cls(model_type=HUB_MODEL_TYPE, latent_dim=int(config.get("latent_dim") or 4096))
    refused = False
    try:
        refuse_if_research_scale(cfg)
    except RuntimeError as exc:
        refused = "load_mina" in str(exc)
    if not refused:
        raise SystemExit("6.8B AutoModel construct was not refused")
    causal = "ForCausalLM" in model_cls.__name__ or "CausalLM" in json.dumps(config)
    shape_ok = all(isinstance(t, torch.Tensor) and t.ndim >= 1 for t in shards.values())
    count_ok = parameters == FROZEN_PARAM_ESTIMATE or n_params == FROZEN_PARAM_ESTIMATE
    report = {
        "path": str(root),
        "load_tensor": bool(shards),
        "n_tensors": len(shards),
        "shape_match": shape_ok,
        "parameter_count": parameters or n_params,
        "parameter_count_ok": count_ok,
        "auto_model": model_cls.__name__,
        "auto_config": config_cls.__name__,
        "auto_model_works_as_type_tag": True,
        "auto_model_for_causal_lm": False,
        "causal_lm_absent": not causal,
        "constructed_6_8b": False,
        "runtime": "load_mina",
        "pass": bool(shards) and shape_ok and count_ok and (not causal) and refused,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit("hf reload gate failed")


if __name__ == "__main__":
    main()
