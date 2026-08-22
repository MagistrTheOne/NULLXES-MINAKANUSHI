"""Export a Hugging Face safetensors mirror. Canonical artifact stays *.mina.

Does not upload. Does not construct 6.8B. Run on H200/B300 after the v0.3
curriculum gate, not on a laptop:

    python scripts/export_safetensors.py \\
      --mina minakanushi_stage0_step128.mina \\
      --out MINAKANUSHI-6.8B
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minakanushi.training.hf_export import DEFAULT_SHARD_BYTES, export_hf_mirror

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_README = ROOT / "models" / "MINA-6.8B" / "HF_README.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mina", type=Path, required=True, help="canonical *.mina")
    parser.add_argument("--out", type=Path, default=Path("MINAKANUSHI-6.8B"))
    parser.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    parser.add_argument(
        "--cards-only",
        action="store_true",
        help="write config/CARD/runtime JSON without converting weights",
    )
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    args = parser.parse_args()
    result = export_hf_mirror(
        args.mina,
        args.out,
        shard_bytes=args.shard_bytes,
        cards_only=args.cards_only,
        readme=args.readme if args.readme.exists() else None,
        license_path=ROOT / "LICENSE",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
