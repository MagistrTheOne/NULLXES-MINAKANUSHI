"""6.8B pre-train stack check. Never constructs MinakanushiSystem from 6.8B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from minakanushi.architecture.config import load_architecture, load_training
from minakanushi.architecture.freeze import (
    FROZEN_AT,
    FROZEN_PARAM_ESTIMATE,
    assert_6_8b_frozen,
    assert_may_construct,
)
from minakanushi.training.parallel import plan_from_training
from minakanushi.training.parameter_inventory import estimate_parameters
from minakanushi.training.shard import merge_tensor_maps, split_tensor_map


def main() -> None:
    parser = argparse.ArgumentParser(description="MINAKANUSHI 6.8B pre-train stack check")
    parser.add_argument("--out", type=Path, default=Path("experiments/gate_6_8b_pretrain"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    arch = load_architecture(root / "configs" / "architecture" / "minakanushi_6_8b.yaml")
    train = load_training(root / "configs" / "training" / "mina_6_8b_sanity.yaml")
    assert_6_8b_frozen(arch)
    n = estimate_parameters(arch)["total_estimate"]
    if n != FROZEN_PARAM_ESTIMATE:
        raise SystemExit(f"parameter formula drifted: {n} != {FROZEN_PARAM_ESTIMATE}")
    plan = plan_from_training(arch, train)
    refused = False
    try:
        assert_may_construct(arch, device="cpu")
    except RuntimeError:
        refused = True
    if not refused:
        raise SystemExit("construct guard failed: CPU 6.8B was allowed")

    linear = nn.Linear(32, 16)
    shards = split_tensor_map(linear.state_dict(), max_bytes=64)
    restored = merge_tensor_maps(shards)
    for key, tensor in linear.state_dict().items():
        if not torch.equal(tensor, restored[key]):
            raise SystemExit(f"shard restore mismatch: {key}")

    report = {
        "gate": "6_8b_pretrain",
        "frozen_at": FROZEN_AT,
        "profile": "minakanushi_6_8b",
        "param_estimate": n,
        "plan": {
            "parallelism": plan.parallelism,
            "sharding": plan.sharding,
            "precision": plan.precision,
            "reduce_dtype": plan.reduce_dtype,
            "activation_checkpoint": plan.activation_checkpoint,
            "cognition_budget": plan.cognition_budget,
            "shard_max_bytes": plan.shard_max_bytes,
        },
        "construct_cpu_refused": True,
        "constructed_6_8b": False,
        "closed": ["yunmu", "gate10", "fp16", "token_dataset", "rtx_6000_train"],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "sanity_stack.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
