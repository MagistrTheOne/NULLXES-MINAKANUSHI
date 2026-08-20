"""Stamp SelfModel passport into a *.mina checkpoint. No training. No 6.8B construct."""

from __future__ import annotations

import argparse
from pathlib import Path

from minakanushi.identity.initialize import initialize_identity


def main() -> None:
    parser = argparse.ArgumentParser(description="MINAKANUSHI Identity Initialization")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("experiments/mina_6_8b_identity/MINA-6.8B-IdentityBound.mina"))
    args = parser.parse_args()
    path = initialize_identity(args.checkpoint, args.out)
    print(f"identity-bound {path}")


if __name__ == "__main__":
    main()
