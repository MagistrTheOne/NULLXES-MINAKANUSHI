"""Auditable constraint decisions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConstraintAudit:
    strategy_id: str
    allowed: bool
    reasons: tuple[str, ...]
