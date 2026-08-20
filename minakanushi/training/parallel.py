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
            activation_checkpoint=bool(train.activation_checkpoint),
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


def is_fsdp_dtensor(parameter: torch.Tensor) -> bool:
    return type(parameter).__name__ == "DTensor"


def replicated_trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    """Trainable weights FSDP2 CognitiveBlock wrap does not own."""
    seen: set[int] = set()
    out: list[nn.Parameter] = []
    for parameter in module.parameters():
        if not parameter.requires_grad or is_fsdp_dtensor(parameter):
            continue
        marker = id(parameter)
        if marker in seen:
            continue
        seen.add(marker)
        out.append(parameter)
    return out


def register_replicated_grad_sync(module: nn.Module, world_size: int | None = None) -> int:
    """Average grads on replicated modules so multi-GPU 6.8B does not diverge.

    Trainer calls perception.encode, memory.hints, and future.predict — not
    FSDPModule.forward. Root fully_shard therefore cannot return: those paths
    mix dense CUDA tensors with DTensor weights. CognitiveBlock stays ZeRO-3.
    Remaining trainable weights stay dense and must all-reduce.
    """
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("replicated grad sync requires torchrun / init_process_group")
    size = int(dist.get_world_size() if world_size is None else world_size)
    if size < 1:
        raise ValueError(f"world_size must be >= 1, got {size}")
    hooked = 0
    for parameter in replicated_trainable_parameters(module):
        _attach_replicated_grad_hook(parameter, size)
        hooked += 1
    return hooked


def _attach_replicated_grad_hook(parameter: nn.Parameter, world_size: int) -> None:
    import torch.distributed as dist

    def _allreduce_grad(grad: torch.Tensor) -> torch.Tensor:
        if world_size <= 1:
            return grad
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        return grad / world_size

    if hasattr(parameter, "register_post_accumulate_grad_hook"):

        def _post_hook(param: torch.Tensor) -> None:
            if param.grad is None or world_size <= 1:
                return
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)

        parameter.register_post_accumulate_grad_hook(_post_hook)
        return
    parameter.register_hook(_allreduce_grad)


def wrap_fsdp2(module: nn.Module, activation_checkpoint: bool | None = None) -> nn.Module:
    """ZeRO-3 on CognitiveBlock plus averaged grads on replicated leftovers.

    Activation checkpoint must wrap the dense block *before* fully_shard.
    Checkpointing inside an already-sharded FSDP module captures DTensor
    all-gathers as saved tensors; recompute then sees a different tensor list
    ([4096, 4096] weights vs [512, 4096] activations).

    Requires torch.distributed already initialized (torchrun).
    """
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("FSDP2 fully_shard requires torchrun / init_process_group")
    try:
        from torch.distributed.fsdp import fully_shard
    except ImportError as exc:
        raise RuntimeError("PyTorch FSDP2 fully_shard is required for 6.8B") from exc

    from minakanushi.core.cognitive_block import CognitiveBlock
    from minakanushi.core.dynamic_world_core import DynamicWorldCore

    world_core = getattr(module, "world_core", None)
    use_ac = bool(world_core.activation_checkpoint) if activation_checkpoint is None else bool(activation_checkpoint)
    if use_ac:
        raise RuntimeError(
            "FSDP2 activation_checkpoint is disabled for 6.8B sanity: "
            "PyTorch checkpoint + fully_shard currently records FSDP all-gather "
            "tensors and fails recompute metadata checks. Set activation_checkpoint: false."
        )
    if isinstance(world_core, DynamicWorldCore):
        new_blocks = []
        for block in world_core.cognitive_blocks():
            block.activation_checkpoint = False
            fully_shard(block)
            new_blocks.append(block)
        world_core.blocks = nn.ModuleList(new_blocks)
        world_core._activation_checkpoint = use_ac
    else:
        for child in module.modules():
            if isinstance(child, CognitiveBlock):
                fully_shard(child)
    # Do not fully_shard MinakanushiSystem. Root wrap turns perception/NPF
    # Linear weights into DTensors; encode() feeds dense CUDA tensors and
    # aten.addmm then raises mixed Tensor/DTensor. ZeRO-3 lives in DWC blocks.
    register_replicated_grad_sync(module)
    return module
