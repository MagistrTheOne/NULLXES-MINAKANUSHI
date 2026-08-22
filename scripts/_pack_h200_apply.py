"""Local helper: pack v0.3.1 publish files into an SSH apply script."""

from __future__ import annotations

import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "scripts/test_hf_reload.py",
    "minakanushi/training/export_roundtrip.py",
    "minakanushi/training/hf_export.py",
    "scripts/publish_v031_hf.py",
    "models/MINA-6.8B/HF_README.md",
)
OUT = Path.home() / "AppData" / "Local" / "Temp" / "mina_h200_apply.sh"


def main() -> None:
    lines = [
        "cd /workspace/NULLXES-MINAKANUSHI",
        "python3 << 'PY'",
        "import base64",
        "from pathlib import Path",
        "FILES = {",
    ]
    for rel in FILES:
        blob = base64.b64encode((ROOT / rel).read_bytes()).decode("ascii")
        lines.append(f"    {rel!r}: {blob!r},")
    lines.extend(
        [
            "}",
            'root = Path("/workspace/NULLXES-MINAKANUSHI")',
            "for rel, blob in FILES.items():",
            "    path = root / rel",
            "    path.parent.mkdir(parents=True, exist_ok=True)",
            "    path.write_bytes(base64.b64decode(blob))",
            '    print("wrote", rel, path.stat().st_size)',
            "PY",
            "exit",
        ]
    )
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT, OUT.stat().st_size)


if __name__ == "__main__":
    main()
