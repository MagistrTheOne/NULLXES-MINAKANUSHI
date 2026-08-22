"""Split mina_6_8b_v03 into train/heldout by (seed, scenario, episode_index).

Does not copy JSON. Does not shuffle files.

    python scripts/split_heldout.py --root dataset/mina_6_8b_v03
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minakanushi.training.heldout import write_heldout_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("dataset/mina_6_8b_v03"))
    args = parser.parse_args()
    report = write_heldout_split(args.root)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
