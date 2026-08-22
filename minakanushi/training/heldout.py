"""Held-out split by trajectory identity. Not a random file shuffle.

Identity is (seed, scenario, episode_index). Same trajectory cannot sit in
both train and held-out. Files stay in physics|agency|causality|embodiment/.
Only index.jsonl copies are written under train/ and heldout/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minakanushi.training.episode_dataset import PHASE_ORDER, scenario_from_episode_id

HELD_OUT_MOD = 10
HELD_OUT_REMAINDER = 9
SKIP_JSON_NAMES = frozenset({"dataset_report.json", "splits.json", "dataset_manifest.json"})
SKIP_DIR_NAMES = frozenset({"train", "heldout"})


def parse_episode_id(episode_id: str) -> tuple[str, int, int]:
    """episode_id is `{scenario}-{seed}-{episode_index}`."""
    scenario, seed_s, idx_s = str(episode_id).rsplit("-", 2)
    return scenario, int(seed_s), int(idx_s)


def is_heldout_index(episode_index: int) -> bool:
    return int(episode_index) % HELD_OUT_MOD == HELD_OUT_REMAINDER


def identity_key(seed: int, scenario: str, episode_index: int) -> tuple[int, str, int]:
    return (int(seed), str(scenario), int(episode_index))


def load_pack_index(root: Path) -> list[dict[str, Any]]:
    root = Path(root)
    index = root / "index.jsonl"
    rows: list[dict[str, Any]] = []
    if index.is_file():
        for line in index.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            path = str(rec["path"])
            episode_id = str(rec.get("episode_id") or Path(path).stem)
            scenario = str(rec.get("scenario") or scenario_from_episode_id(episode_id))
            seed = rec.get("seed")
            episode_index = rec.get("episode_index")
            if seed is None or episode_index is None:
                scenario, seed, episode_index = parse_episode_id(episode_id)
            rows.append(
                {
                    "path": path,
                    "phase": str(rec.get("phase") or Path(path).parent.name),
                    "episode_id": episode_id,
                    "scenario": scenario,
                    "seed": int(seed),
                    "episode_index": int(episode_index),
                }
            )
        return rows
    for path in sorted(root.rglob("*.json")):
        if path.name in SKIP_JSON_NAMES or path.parent.name in SKIP_DIR_NAMES:
            continue
        episode_id = path.stem
        scenario, seed, episode_index = parse_episode_id(episode_id)
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "phase": path.parent.name if path.parent.name in PHASE_ORDER else "physics",
                "episode_id": episode_id,
                "scenario": scenario,
                "seed": seed,
                "episode_index": episode_index,
            }
        )
    return rows


def write_heldout_split(root: str | Path) -> dict[str, Any]:
    """Write train/index.jsonl and heldout/index.jsonl. Does not copy episode JSON."""
    root = Path(root)
    rows = load_pack_index(root)
    if not rows:
        raise FileNotFoundError(f"no episodes to split under {root}")
    identities = [identity_key(r["seed"], r["scenario"], r["episode_index"]) for r in rows]
    train_rows = [r for r in rows if not is_heldout_index(r["episode_index"])]
    held_rows = [r for r in rows if is_heldout_index(r["episode_index"])]
    overlap = {
        identity_key(r["seed"], r["scenario"], r["episode_index"]) for r in train_rows
    } & {identity_key(r["seed"], r["scenario"], r["episode_index"]) for r in held_rows}
    if overlap:
        raise ValueError(f"train/held-out identity leak: {sorted(overlap)[:3]}")

    def _write(split: str, chosen: list[dict[str, Any]]) -> Path:
        dest = root / split
        dest.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(
                {
                    "path": r["path"],
                    "phase": r["phase"],
                    "episode_id": r["episode_id"],
                    "scenario": r["scenario"],
                    "seed": r["seed"],
                    "episode_index": r["episode_index"],
                    "split": split,
                },
                sort_keys=True,
            )
            for r in chosen
        ]
        path = dest / "index.jsonl"
        path.write_text(("\n".join(lines) + ("\n" if lines else "")), encoding="utf-8")
        return path

    train_path = _write("train", train_rows)
    held_path = _write("heldout", held_rows)
    report = {
        "dataset_root": str(root),
        "rule": f"episode_index % {HELD_OUT_MOD} == {HELD_OUT_REMAINDER} -> heldout",
        "identity": ["seed", "scenario", "episode_index"],
        "not": "random file shuffle",
        "n_episodes": len(rows),
        "n_train": len(train_rows),
        "n_heldout": len(held_rows),
        "n_identities": len(set(identities)),
        "train_index": str(train_path),
        "heldout_index": str(held_path),
        "leak": False,
    }
    (root / "splits.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
