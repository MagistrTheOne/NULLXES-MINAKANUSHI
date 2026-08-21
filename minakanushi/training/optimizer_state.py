"""Load H200 integer-id Adam state onto FSDP2-wrapped 6.8B.

Status Core `*.mina` stores optimizer as `state[28]['step']` (or flattened
`state.28.step`). FSDP2 `set_optimizer_state_dict` unflattens via an FQN map
and raises KeyError (`state.28.step`, then `perception.vector.net.0.weight`
if those ids are rewritten to dotted FQNs).

Weights still go through `set_model_state_dict`. Optimizer moments are placed
onto `optimizer.param_groups` in constructor order — the same run, not a clone.
"""

from __future__ import annotations

import re
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

_INTEGER_FLAT_STATE = re.compile(r"^state\.(\d+)\.(.+)$")
_INTEGER_FLAT_GROUP = re.compile(r"^param_groups\.(\d+)\.(.+)$")
_MOMENT_KEYS = frozenset({"exp_avg", "exp_avg_sq", "max_exp_avg_sq"})


def optimizer_params(optimizer: Optimizer) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for group in optimizer.param_groups:
        for param in group["params"]:
            params.append(param)
    if not params:
        raise ValueError("optimizer has no parameters")
    return params


def integer_optimizer_state(opt_state: Any) -> dict[str, Any]:
    """Nested `{state, param_groups}` with integer param ids."""
    if opt_state is None:
        raise ValueError("checkpoint has no optimizer state; refusing silent resume")
    if not isinstance(opt_state, dict):
        raise ValueError(f"optimizer state must be a dict, got {type(opt_state).__name__}")
    return _maybe_unflatten_integer_optimizer(opt_state)


def apply_optimizer_checkpoint(optimizer: Optimizer, opt_state: Any) -> None:
    """Write checkpoint Adam moments onto the live optimizer. Fail loud on shape/id mismatch."""
    nested = integer_optimizer_state(opt_state)
    raw_state = nested.get("state") or {}
    params = optimizer_params(optimizer)
    if not raw_state:
        return
    sample = next(iter(raw_state))
    if isinstance(sample, str) and not str(sample).isdigit():
        raise ValueError(
            f"optimizer state is FQN-keyed ({sample!r}); FSDP2 resume expects integer ids from H200 *.mina"
        )
    for key, fields in raw_state.items():
        index = int(key)
        if index < 0 or index >= len(params):
            raise ValueError(
                f"optimizer state index {index} out of range for {len(params)} parameters; "
                "refusing silent resume"
            )
        param = params[index]
        if not isinstance(fields, dict):
            raise ValueError(f"optimizer state[{index}] must be a dict")
        placed: dict[str, Any] = {}
        for name, value in fields.items():
            if name in _MOMENT_KEYS and torch.is_tensor(value):
                placed[name] = _moment_like_param(value, param, index=index, name=name)
            elif torch.is_tensor(value):
                placed[name] = value.detach().to(device=param.device)
            else:
                placed[name] = value
        optimizer.state[param] = placed
    groups = nested.get("param_groups") or []
    if groups:
        src = groups[0]
        for group in optimizer.param_groups:
            for field in ("lr", "weight_decay", "betas", "eps", "amsgrad"):
                if field in src:
                    group[field] = src[field]


def _moment_like_param(value: torch.Tensor, param: nn.Parameter, *, index: int, name: str) -> torch.Tensor:
    source = value.detach()
    if type(source).__name__ == "DTensor" and hasattr(source, "full_tensor"):
        source = source.full_tensor()
    if type(param).__name__ == "DTensor":
        from torch.distributed.tensor import distribute_tensor

        full = source.to(device=param.device, dtype=param.dtype)
        if tuple(full.shape) != tuple(param.shape):
            raise ValueError(
                f"optimizer {name} index {index} shape {tuple(full.shape)} != DTensor param {tuple(param.shape)}"
            )
        return distribute_tensor(full, param.device_mesh, param.placements)
    out = source.to(device=param.device, dtype=param.dtype)
    if tuple(out.shape) != tuple(param.shape):
        raise ValueError(
            f"optimizer {name} index {index} shape {tuple(out.shape)} != param {tuple(param.shape)}"
        )
    return out


def _maybe_unflatten_integer_optimizer(opt_state: dict[str, Any]) -> dict[str, Any]:
    if "state" in opt_state and "param_groups" in opt_state:
        groups = opt_state["param_groups"] or []
        return {
            "state": dict(opt_state["state"]),
            "param_groups": [dict(group) for group in groups],
        }
    integer_flat = any(_INTEGER_FLAT_STATE.match(str(key)) for key in opt_state)
    if not integer_flat:
        raise ValueError(
            "optimizer payload is neither nested {state,param_groups} nor integer-flat "
            f"(keys like state.28.step); keys={list(opt_state)[:8]!r}"
        )
    state: dict[int, dict[str, Any]] = {}
    groups: dict[int, dict[str, Any]] = {}
    for key, value in opt_state.items():
        match = _INTEGER_FLAT_STATE.match(str(key))
        if match:
            index = int(match.group(1))
            state.setdefault(index, {})[match.group(2)] = value
            continue
        match = _INTEGER_FLAT_GROUP.match(str(key))
        if match:
            gi = int(match.group(1))
            rest = match.group(2)
            group = groups.setdefault(gi, {})
            if rest.startswith("params."):
                slot = int(rest.split(".", 1)[1])
                params = group.setdefault("params", {})
                if not isinstance(params, dict):
                    raise ValueError("param_groups params flattened inconsistently")
                params[slot] = value
            else:
                group[rest] = value
            continue
        raise ValueError(f"unrecognized flattened optimizer key {key!r}")
    param_groups = []
    if groups:
        param_groups = [_coerce_group_params(groups[i]) for i in range(len(groups))]
    return {"state": state, "param_groups": param_groups}


def _coerce_group_params(group: dict[str, Any]) -> dict[str, Any]:
    out = dict(group)
    params = out.get("params")
    if isinstance(params, dict):
        out["params"] = [params[i] for i in range(len(params))]
    return out
