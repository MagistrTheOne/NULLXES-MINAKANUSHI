"""Canonical episode dump for replay identity checks."""

from __future__ import annotations

import json
from typing import Any


def canonical_json(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def records_identical(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return canonical_json(a) == canonical_json(b)
