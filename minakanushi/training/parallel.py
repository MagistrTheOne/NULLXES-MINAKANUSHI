"""6.8B parallelism contract: FSDP2 / ZeRO-3, bf16 compute, fp32 reductions.

Does not construct MinakanushiSystem. Wrap is for H200/B300 torchrun only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from minakanushi.architecture.config import ArchitectureConfig, TrainingConfig
from minakanushi.architecture.freeze import is_6_8b_profile

ALLOWED_PRECISION = "bf16"
FORBIDDEN_PRECISION = ("fp16", "float16")


@dataclass(frozen=True)
class TrainingStackPlan:
    parallelism: str
    sharding: str
    precision: str
    reduce_dtype: str
    activation_checkpoint: bool
    cognition_budget: int
    shard_max_bytes: int


def plan_from_training(arch: ArchitectureConfig, train: TrainingConfig) -> TrainingStackPlan:
    precision = str(train.precision).lower()
    if is_6_8b_profile(arch):
        if precision in FORBIDDEN_PRECISION:
            raise ValueError("6.8B forbids FP16; use bf16")
        if precision not in {ALLOWED_PRECISION, "bfloat16"}:
            raise ValueError(f"6.8B sanity/train requires bf16, got {train.precision}")
        parallelism = train.parallelism if train.parallelism != "none" else "fsdp2_zero3"
        if parallelism != "fsdp2_zero3":
            raise ValueError(f"6.8B requires fsdp2_zero3, got {parallelism}")
        return TrainingStackPlan(
            parallelism="fsdp2_zero3",
            sharding="fully_shard",
            precision="bf16",
            reduce_dtype="fp32",
            activation_checkpoint=True,
            cognition_budget=int(arch.cognition.budget),
            shard_max_bytes=int(train.shard_max_bytes) if train.shard_max_bytes > 0 else 1_073_741_824,
        )
    return TrainingStackPlan(
        parallelism=str(train.parallelism),
        sharding="none",
        precision=precision,
        reduce_dtype="fp32",
        activation_checkpoint=bool(train.activation_checkpoint),
        cognition_budget=int(arch.cognition.budget),
        shard_max_bytes=int(train.shard_max_bytes),
    )


def wrap_fsdp2(module: nn.Module) -> nn.Module:
    """ZeRO-3 equivalent: fully shard params, grads, optimizer state.

    Requires torch.distributed already initialized (torchrun).
    """
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("FSDP2 fully_shard requires torchrun / init_process_group")
    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    except ImportError as exc:
        raise RuntimeError("PyTorch FSDP2 fully_shard is required for 6.8B") from exc

    from minakanushi.core.cognitive_block import CognitiveBlock

    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for child in module.modules():
        if isinstance(child, CognitiveBlock):
            fully_shard(child, mp_policy=mp)
    fully_shard(module, mp_policy=mp)
    return module
