"""Inspect one SyntheticWorld episode JSON. Not training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from simulations.synthetic_world.inspector import format_episode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--max-frames", type=int, default=8)
    args = parser.parse_args()
    record = json.loads(args.path.read_text(encoding="utf-8"))
    print(format_episode(record, max_frames=args.max_frames), end="")


if __name__ == "__main__":
    main()
