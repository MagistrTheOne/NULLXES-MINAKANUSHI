"""6.8B parallelism contract: FSDP2 / ZeRO-3, bf16 compute, fp32 reductions.

Does not construct MinakanushiSystem. Wrap is for H200/B300 torchrun only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

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


def dist_is_initialized() -> bool:
    import torch.distributed as dist

    return bool(dist.is_available() and dist.is_initialized())


def dist_rank() -> int:
    import torch.distributed as dist

    if dist_is_initialized():
        return int(dist.get_rank())
    return 0


def is_rank0() -> bool:
    return dist_rank() == 0


def dist_barrier() -> None:
    import torch.distributed as dist

    if dist_is_initialized():
        dist.barrier()


def is_fsdp_wrapped(module: nn.Module) -> bool:
    for parameter in module.parameters():
        if type(parameter).__name__ == "DTensor":
            return True
    for child in module.modules():
        name = type(child).__name__
        if name == "FSDPModule" or name.startswith("FullySharded"):
            return True
    return False


def torchrun_env_present() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ


def init_process_group_if_needed(parallelism: str, device: str) -> None:
    """Bind LOCAL_RANK and init the process group. torchrun only sets env vars."""
    import torch.distributed as dist

    needs = str(parallelism) == "fsdp2_zero3" or torchrun_env_present()
    if not needs:
        return
    if not torchrun_env_present():
        raise RuntimeError(
            "fsdp2_zero3 requires torchrun (RANK, WORLD_SIZE, LOCAL_RANK). "
            "python scripts/train.py is not enough."
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("fsdp2_zero3 requested CUDA but torch.cuda.is_available() is False")
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"
    if dist.is_initialized():
        return
    dist.init_process_group(backend=backend)


def training_device(requested: str) -> torch.device:
    """Map training.device through LOCAL_RANK so ranks do not all pin to cuda:0."""
    if requested == "cpu" or requested.startswith("cpu"):
        return torch.device("cpu")
    if not str(requested).startswith("cuda"):
        raise ValueError(f"unsupported device '{requested}'")
    if not torch.cuda.is_available():
        raise RuntimeError("config requested cuda but torch.cuda.is_available() is False")
    local = os.environ.get("LOCAL_RANK")
    if local is None:
        if ":" in requested:
            return torch.device(requested)
        return torch.device("cuda")
    index = int(local)
    torch.cuda.set_device(index)
    return torch.device(f"cuda:{index}")


def _cpu_state_dict(module: nn.Module) -> dict[str, Any]:
    return {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in module.state_dict().items()}


def collect_full_checkpoint(module: nn.Module, optimizer: Optimizer | None) -> dict[str, Any] | None:
    """All ranks must call this.

    FSDP2 local state_dict() is a shard. Gather a full CPU copy, then only rank 0
    returns a payload for *.mina. Other ranks return None.
    """
    if is_fsdp_wrapped(module):
        try:
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_model_state_dict,
                get_optimizer_state_dict,
            )
        except ImportError as exc:
            raise RuntimeError("FSDP2 checkpoint gather requires torch.distributed.checkpoint.state_dict") from exc
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        system = get_model_state_dict(module, options=options)
        opt_state = None
        if optimizer is not None:
            opt_state = get_optimizer_state_dict(module, optimizer, options=options)
        dist_barrier()
        if not is_rank0():
            return None
        total = 0
        for value in system.values():
            if torch.is_tensor(value):
                total += int(value.numel())
        return {
            "system": system,
            "optimizer": opt_state,
            "parameter_report": {"total": int(total), "trainable": int(total)},
            "runtime": None,
            "gathered": True,
        }
    payload = {
        "system": _cpu_state_dict(module),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "parameter_report": module.parameter_report() if hasattr(module, "parameter_report") else None,
        "runtime": None,
        "gathered": False,
    }
    dist_barrier()
    if not is_rank0():
        return None
    return payload


def apply_full_checkpoint(module: nn.Module, optimizer: Optimizer | None, payload: dict[str, Any]) -> None:
    """Restore a gathered *.mina payload. Collective when the module is FSDP-wrapped."""
    system_state = payload["system"]
    if is_fsdp_wrapped(module):
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
            set_optimizer_state_dict,
        )

        options = StateDictOptions(full_state_dict=True)
        set_model_state_dict(module, model_state_dict=system_state, options=options)
        if optimizer is not None:
            if payload.get("optimizer") is None:
                raise ValueError("checkpoint has no optimizer state; refusing silent resume")
            set_optimizer_state_dict(
                module,
                optimizer,
                optim_state_dict=payload["optimizer"],
                options=options,
            )
        return
    incompatible = module.load_state_dict(system_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"strict load failed: {incompatible}")
    if optimizer is not None:
        if payload.get("optimizer") is None:
            raise ValueError("checkpoint has no optimizer state; refusing silent resume")
        optimizer.load_state_dict(payload["optimizer"])


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
    # Do not fully_shard MinakanushiSystem. Root wrap turns perception/NPF
    # Linear weights into DTensors; encode() feeds dense CUDA tensors and
    # aten.addmm then raises mixed Tensor/DTensor. ZeRO-3 lives in DWC blocks.
    return module
