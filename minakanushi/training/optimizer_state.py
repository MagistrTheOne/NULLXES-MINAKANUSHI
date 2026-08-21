"""Normalize Adam optimizer payloads for FSDP2 resume.

H200 Status Core saved full optimizer state keyed by integer param ids
(`state.28.step` after flatten). FSDP2 `set_optimizer_state_dict` builds an
FQN mapping and raises KeyError on those ids. Weights-only load would be a
clone. This remaps ids → named_parameters FQNs so resume stays the same run.
"""

from __future__ import annotations

import re
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

_INTEGER_FLAT_STATE = re.compile(r"^state\.(\d+)\.(.+)$")
_INTEGER_FLAT_GROUP = re.compile(r"^param_groups\.(\d+)\.(.+)$")


def optimizer_param_fqns(module: nn.Module, optimizer: Optimizer) -> tuple[str, ...]:
    """FQNs in AdamW constructor order (`module.parameters()`)."""
    id_to_name = {id(param): name for name, param in module.named_parameters()}
    names: list[str] = []
    for group in optimizer.param_groups:
        for param in group["params"]:
            name = id_to_name.get(id(param))
            if name is None:
                raise ValueError("optimizer param is not in module.named_parameters(); refusing silent resume")
            names.append(name)
    if not names:
        raise ValueError("optimizer has no parameters")
    return tuple(names)


def normalize_optimizer_state(
    module: nn.Module,
    optimizer: Optimizer,
    opt_state: Any,
) -> dict[str, Any]:
    """Return FQN-keyed optimizer state for FSDP2 set_optimizer_state_dict."""
    if opt_state is None:
        raise ValueError("checkpoint has no optimizer state; refusing silent resume")
    if not isinstance(opt_state, dict):
        raise ValueError(f"optimizer state must be a dict, got {type(opt_state).__name__}")
    nested = _maybe_unflatten_integer_optimizer(opt_state)
    fqns = optimizer_param_fqns(module, optimizer)
    id_to_name = {id(param): name for name, param in module.named_parameters()}
    return _remap_integer_state_to_fqn(nested, fqns, id_to_name=id_to_name)


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


def _remap_integer_state_to_fqn(
    opt_state: dict[str, Any],
    fqns: tuple[str, ...],
    *,
    id_to_name: dict[int, str] | None = None,
) -> dict[str, Any]:
    raw_state = opt_state.get("state") or {}
    groups = [
        _rewrite_group_params(group, fqns, id_to_name=id_to_name)
        for group in opt_state.get("param_groups") or []
    ]
    if not raw_state:
        return {"state": {}, "param_groups": groups}
    sample = next(iter(raw_state))
    if isinstance(sample, str) and not str(sample).isdigit():
        return {"state": dict(raw_state), "param_groups": groups}

    remapped: dict[str, Any] = {}
    for key, value in raw_state.items():
        index = int(key)
        if index < 0 or index >= len(fqns):
            raise ValueError(
                f"optimizer state index {index} out of range for {len(fqns)} parameters; "
                "refusing silent resume"
            )
        remapped[fqns[index]] = value
    if not groups:
        groups = [{"params": list(fqns)}]
    return {"state": remapped, "param_groups": groups}


def _rewrite_group_params(
    group: dict[str, Any],
    fqns: tuple[str, ...],
    *,
    id_to_name: dict[int, str] | None = None,
) -> dict[str, Any]:
    out = {k: v for k, v in group.items() if k != "params"}
    params = group.get("params")
    if params is None:
        out["params"] = list(fqns)
        return out
    rewritten: list[str] = []
    for item in params:
        if isinstance(item, str) and not item.isdigit():
            rewritten.append(item)
            continue
        if torch.is_tensor(item) or isinstance(item, nn.Parameter):
            if not id_to_name:
                raise ValueError("optimizer param_groups still hold live tensors; refusing silent resume")
            name = id_to_name.get(id(item))
            if name is None:
                raise ValueError("optimizer param_groups tensor is not in module.named_parameters()")
            rewritten.append(name)
            continue
        index = int(item)
        if index < 0 or index >= len(fqns):
            raise ValueError(f"param_groups index {index} out of range for {len(fqns)} parameters")
        rewritten.append(fqns[index])
    out["params"] = rewritten
    return out
