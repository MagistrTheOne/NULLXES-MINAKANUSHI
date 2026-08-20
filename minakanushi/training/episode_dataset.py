"""Stream JSON episodes. Source of truth: dataset/mina_6_8b. Not a preload."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from simulations.synthetic_world.dataset import Episode
from simulations.synthetic_world.dataset_v1 import record_to_episode, validate_episode_record

PHASE_ORDER = ("physics", "agency", "causality", "embodiment")


def scenario_from_episode_id(episode_id: str) -> str:
    """episode_id is `{scenario}-{seed}-{episode_index}`."""
    parts = str(episode_id).rsplit("-", 2)
    if len(parts) != 3:
        raise ValueError(f"cannot parse scenario from episode_id={episode_id!r}")
    return parts[0]


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
        self.paths, self.scenarios = self._index_rows()
        if not self.paths:
            raise FileNotFoundError(f"no episode JSON under {self.root}")

    def _index_rows(self) -> tuple[tuple[Path, ...], tuple[str, ...]]:
        index = self.root / "index.jsonl"
        rows: list[tuple[Path, str]] = []
        if index.is_file():
            for line in index.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                path = self.root / rec["path"]
                scenario = str(rec["scenario"]) if rec.get("scenario") else scenario_from_episode_id(path.stem)
                rows.append((path, scenario))
            return tuple(p for p, _ in rows), tuple(s for _, s in rows)
        found = sorted(self.root.rglob("*.json"))
        found = [p for p in found if p.name not in {"dataset_report.json", "index.jsonl"}]
        rng = np.random.default_rng(self.seed)
        order = np.arange(len(found))
        rng.shuffle(order)
        ordered = tuple(found[int(i)] for i in order)
        return ordered, tuple(scenario_from_episode_id(path.stem) for path in ordered)

    def __len__(self) -> int:
        return len(self.paths)

    def record(self, index: int) -> dict:
        path = self.paths[int(index) % len(self.paths)]
        raw = json.loads(path.read_text(encoding="utf-8"))
        validate_episode_record(raw, curriculum_6_8b=self.curriculum_6_8b)
        return raw

    def episode(self, index: int) -> Episode:
        return record_to_episode(self.record(index), curriculum_6_8b=self.curriculum_6_8b)

    def episode_for_scenario(self, scenario: str, episode_index: int = 0) -> Episode:
        hits = [i for i, name in enumerate(self.scenarios) if name == scenario]
        if not hits:
            raise FileNotFoundError(f"no JSON episode with scenario={scenario!r} under {self.root}")
        return self.episode(hits[int(episode_index) % len(hits)])
