"""Stream JSON episodes. Source of truth: dataset/mina_6_8b. Not a preload."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from simulations.synthetic_world.dataset import Episode
from simulations.synthetic_world.dataset_v1 import record_to_episode, validate_episode_record

PHASE_ORDER = ("physics", "agency", "causality", "embodiment")


class JsonEpisodeDataset:
    def __init__(
        self,
        root: str | Path,
        *,
        seed: int = 7,
        curriculum_6_8b: bool = True,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"dataset root missing: {self.root}")
        self.seed = int(seed)
        self.curriculum_6_8b = bool(curriculum_6_8b)
        self.paths = self._index_paths()
        if not self.paths:
            raise FileNotFoundError(f"no episode JSON under {self.root}")

    def _index_paths(self) -> tuple[Path, ...]:
        index = self.root / "index.jsonl"
        if index.is_file():
            rows = []
            for line in index.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                rows.append(self.root / rec["path"])
            return tuple(rows)
        found = sorted(self.root.rglob("*.json"))
        found = [p for p in found if p.name not in {"dataset_report.json", "index.jsonl"}]
        rng = np.random.default_rng(self.seed)
        order = np.arange(len(found))
        rng.shuffle(order)
        return tuple(found[int(i)] for i in order)

    def __len__(self) -> int:
        return len(self.paths)

    def record(self, index: int) -> dict:
        path = self.paths[int(index) % len(self.paths)]
        raw = json.loads(path.read_text(encoding="utf-8"))
        validate_episode_record(raw, curriculum_6_8b=self.curriculum_6_8b)
        return raw

    def episode(self, index: int) -> Episode:
        return record_to_episode(self.record(index), curriculum_6_8b=self.curriculum_6_8b)
