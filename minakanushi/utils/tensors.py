"""Finite-tensor guards and documented boundary checks."""

from __future__ import annotations

import torch
from torch import Tensor


def assert_finite(name: str, tensor: Tensor) -> None:
    if not torch.isfinite(tensor).all():
        n_nan = int(torch.isnan(tensor).sum().item())
        n_inf = int(torch.isinf(tensor).sum().item())
        raise ValueError(f"{name} contains non-finite values: nan={n_nan} inf={n_inf} shape={tuple(tensor.shape)}")


def assert_shape(name: str, tensor: Tensor, expected: tuple[int, ...]) -> None:
    actual = tuple(tensor.shape)
    if len(actual) != len(expected):
        raise ValueError(f"{name} rank mismatch: expected {expected}, got {actual}")
    for a, e in zip(actual, expected):
        if e != -1 and a != e:
            raise ValueError(f"{name} shape mismatch: expected {expected}, got {actual}")


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("config requested cuda but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported device '{name}'")


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if name not in mapping:
        raise ValueError(f"unsupported precision '{name}'")
    return mapping[name]
