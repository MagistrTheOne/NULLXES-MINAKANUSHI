"""v0.3.1 dataset pack: CPU creates, H200 only verifies. Never constructs 6.8B."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from minakanushi.architecture.config import load_simulation
from minakanushi.strategy.engine import StrategyEngine
from minakanushi.training.curriculum_audit import audit_curriculum
from minakanushi.training.heldout import (
    HELD_OUT_MOD,
    HELD_OUT_REMAINDER,
    identity_key,
    is_heldout_index,
    load_pack_index,
    parse_episode_id,
    write_heldout_split,
)
from simulations.synthetic_world.curriculum_6_8b import PHASE_LENGTHS, PHASE_ORDER, write_curriculum

ROOT = Path(__file__).resolve().parents[2]
GENERATOR_VERSION = "v0.3.1"
DATASET_NAME = "mina_6_8b_v03"
READY_NAME = ".READY_V031"
MANIFEST_NAME = "dataset_manifest.json"
REPORT_NAME = "dataset_report.json"
PROFILES = ("v031", "cpu_dev")
V031_MIN_N = 250
CPU_DEV_MIN_N = 10
CPU_DEV_LENGTH = 8
PRODUCTION_COUNTS = {"physics": 250, "agency": 250, "causality": 250, "embodiment": 250}
PRODUCTION_SPLIT = {"episodes": 1000, "train": 900, "heldout": 100}
FORBIDDEN_PWM_KEYS = frozenset(
    {"motor_left", "motor_right", "servo", "joint_pwm", "pwm_duty", "raw_pwm", "pwm_left", "pwm_right"}
)
ALLOWED_ACTION_KEYS = frozenset(
    {
        "objective",
        "target",
        "strategy_id",
        "target_state",
        "parameters",
        "confidence",
        "valid_until",
        "abort_conditions",
        "provenance",
        "extras",
    }
)
ALLOWED_OBJECTIVES = frozenset(StrategyEngine.VOCABULARY) | {"AVOID"}


class DatasetContractError(RuntimeError):
    def __init__(self, failures: list[str]) -> None:
        self.failures = list(failures)
        super().__init__("FAIL DATASET CONTRACT: " + "; ".join(self.failures))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_sha(repo: Path | None = None) -> str:
    repo = repo or ROOT
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _walk_pwm(node: Any, *, path: str, hits: list[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else str(key)
            if str(key) in FORBIDDEN_PWM_KEYS:
                hits.append(here)
            if str(key) == "pwm" and value is True:
                hits.append(here)
            _walk_pwm(value, path=here, hits=hits)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _walk_pwm(item, path=f"{path}[{i}]", hits=hits)


def _action_failures(record: dict[str, Any], source: str) -> list[str]:
    failed: list[str] = []
    for i, action in enumerate(record.get("actions") or []):
        if not isinstance(action, dict):
            failed.append(f"{source} actions[{i}] not object")
            continue
        extra = set(action) - ALLOWED_ACTION_KEYS
        if extra & FORBIDDEN_PWM_KEYS:
            failed.append(f"{source} actions[{i}] pwm keys {sorted(extra & FORBIDDEN_PWM_KEYS)}")
        elif extra:
            failed.append(f"{source} actions[{i}] unknown keys {sorted(extra)}")
        objective = str(action.get("objective") or "")
        if objective not in ALLOWED_OBJECTIVES:
            failed.append(f"{source} actions[{i}] objective={objective!r} not ActionIntent")
    for i, step in enumerate(record.get("transitions") or []):
        if not isinstance(step, dict):
            continue
        action = step.get("action_t")
        if isinstance(action, dict):
            extra = set(action) - ALLOWED_ACTION_KEYS
            if extra & FORBIDDEN_PWM_KEYS:
                failed.append(f"{source} transitions[{i}].action_t pwm keys")
            objective = str(action.get("objective") or "")
            if objective and objective not in ALLOWED_OBJECTIVES:
                failed.append(f"{source} transitions[{i}].action_t objective={objective!r}")
    return failed


def _load_episodes(root: Path) -> list[dict[str, Any]]:
    rows = load_pack_index(root)
    records: list[dict[str, Any]] = []
    for row in rows:
        path = root / row["path"]
        rec = json.loads(path.read_text(encoding="utf-8"))
        rec["_path"] = str(path)
        rec["_index"] = row
        records.append(rec)
    return records


def _index_identities(path: Path) -> list[tuple[int, str, int]]:
    keys: list[tuple[int, str, int]] = []
    if not path.is_file():
        return keys
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        episode_id = str(rec.get("episode_id") or Path(str(rec["path"])).stem)
        scenario = str(rec.get("scenario") or "")
        seed = rec.get("seed")
        episode_index = rec.get("episode_index")
        if seed is None or episode_index is None or not scenario:
            scenario, seed, episode_index = parse_episode_id(episode_id)
        keys.append(identity_key(int(seed), str(scenario), int(episode_index)))
    return keys


def _structure_failures(root: Path) -> list[str]:
    failed: list[str] = []
    for phase in PHASE_ORDER:
        if not (root / phase).is_dir():
            failed.append(f"missing dir {phase}/")
    for rel in ("index.jsonl", "train/index.jsonl", "heldout/index.jsonl", "splits.json"):
        if not (root / rel).is_file():
            failed.append(f"missing {rel}")
    return failed


def _split_failures(root: Path, *, profile: str) -> list[str]:
    failed: list[str] = []
    train_keys = set(_index_identities(root / "train" / "index.jsonl"))
    held_keys = set(_index_identities(root / "heldout" / "index.jsonl"))
    leak = train_keys & held_keys
    if leak:
        sample = sorted(leak)[:3]
        failed.append(f"train/heldout identity leak {sample}")
    for seed, scenario, episode_index in held_keys:
        if not is_heldout_index(episode_index):
            failed.append(
                f"heldout {scenario}-{seed}-{episode_index} breaks "
                f"episode_index % {HELD_OUT_MOD} == {HELD_OUT_REMAINDER}"
            )
    for seed, scenario, episode_index in train_keys:
        if is_heldout_index(episode_index):
            failed.append(f"train contains heldout index {scenario}-{seed}-{episode_index}")
    if profile == "v031":
        if len(train_keys) != PRODUCTION_SPLIT["train"]:
            failed.append(f"train={len(train_keys)} want {PRODUCTION_SPLIT['train']}")
        if len(held_keys) != PRODUCTION_SPLIT["heldout"]:
            failed.append(f"heldout={len(held_keys)} want {PRODUCTION_SPLIT['heldout']}")
    elif not held_keys:
        failed.append("heldout empty (cpu_dev needs n>=10 so episode_index 9 exists)")
    return failed


def _length_failures(records: list[dict[str, Any]], *, profile: str) -> list[str]:
    if profile != "v031":
        return []
    failed: list[str] = []
    for rec in records:
        phase = str(rec.get("phase") or "")
        want = PHASE_LENGTHS.get(phase)
        got = len(rec.get("transitions") or [])
        obs = len(rec.get("observations") or [])
        if want is None:
            failed.append(f"{rec.get('_path')} unknown phase {phase!r}")
            continue
        if got != want or obs != want:
            failed.append(f"{phase} length transitions={got} observations={obs} want {want}")
            break
    return failed


def _correction_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_phase: dict[str, int] = {phase: 0 for phase in PHASE_ORDER}
    per_scenario: dict[str, int] = {}
    total = 0
    for rec in records:
        n = len(rec.get("corrections") or [])
        total += n
        phase = str(rec.get("phase") or "missing")
        per_phase[phase] = int(per_phase.get(phase) or 0) + n
        scenario = str(rec.get("scenario") or "missing")
        per_scenario[scenario] = int(per_scenario.get(scenario) or 0) + n
    return {"total": total, "per_phase": per_phase, "per_scenario": per_scenario}


def inspect_pack(root: Path, *, profile: str) -> dict[str, Any]:
    root = Path(root)
    failures = _structure_failures(root)
    records: list[dict[str, Any]] = []
    if not failures:
        records = _load_episodes(root)
        failures.extend(_split_failures(root, profile=profile))
        failures.extend(_length_failures(records, profile=profile))
    phase_counts = {phase: 0 for phase in PHASE_ORDER}
    pwm_hits: list[str] = []
    action_hits: list[str] = []
    for rec in records:
        phase = str(rec.get("phase") or "missing")
        phase_counts[phase] = int(phase_counts.get(phase) or 0) + 1
        source = str(rec.get("_path") or "")
        _walk_pwm(rec, path=source, hits=pwm_hits)
        action_hits.extend(_action_failures(rec, source))
    if pwm_hits:
        failures.append(f"pwm contract {pwm_hits[:8]}")
    if action_hits:
        failures.append(f"ActionIntent schema {action_hits[:8]}")
    corrections = _correction_stats(records)
    n = len(records)
    if profile == "v031":
        if n != PRODUCTION_SPLIT["episodes"]:
            failures.append(f"episodes={n} want {PRODUCTION_SPLIT['episodes']}")
        for phase, want in PRODUCTION_COUNTS.items():
            got = int(phase_counts.get(phase) or 0)
            if got != want:
                failures.append(f"{phase}={got} want {want}")
        if int(corrections["per_phase"].get("causality") or 0) <= 0:
            failures.append("causality corrections == 0")
        if int(corrections["per_phase"].get("embodiment") or 0) <= 0:
            failures.append("embodiment corrections == 0")
    hashes = {}
    for rel in ("index.jsonl", "train/index.jsonl", "heldout/index.jsonl"):
        path = root / rel
        hashes[rel] = sha256_file(path) if path.is_file() else "MISSING"
    return {
        "root": str(root),
        "profile": profile,
        "n_episodes": n,
        "phase_counts": phase_counts,
        "corrections": corrections,
        "pwm": False if not pwm_hits else True,
        "hashes": hashes,
        "failures": failures,
        "pass": not failures,
    }


def _write_manifest(
    root: Path,
    *,
    profile: str,
    seed: int,
    inspection: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    lengths = dict(PHASE_LENGTHS) if profile == "v031" else {"cpu_dev": CPU_DEV_LENGTH}
    if profile == "v031":
        failed_audit = [name for name, ok in (audit.get("gate") or {}).items() if not ok]
        if failed_audit:
            inspection["failures"] = list(inspection["failures"]) + [f"audit:{name}" for name in failed_audit]
            inspection["pass"] = False
    audit_pass = "PASS" if inspection["pass"] else "FAIL"
    manifest = {
        "dataset": DATASET_NAME,
        "training_cycle": "v0.3.1",
        "profile": profile,
        "seed": int(seed),
        "episodes": int(inspection["n_episodes"]),
        "train": len(_index_identities(root / "train" / "index.jsonl")),
        "heldout": len(_index_identities(root / "heldout" / "index.jsonl")),
        "phase_counts": inspection["phase_counts"],
        "transition_lengths": lengths,
        "pwm": False,
        "audit": audit_pass,
        "corrections": inspection["corrections"],
        "hashes": inspection["hashes"],
        "source_of_truth": "NULLXES SyntheticWorld",
        "hf_data": False,
        "generator_version": GENERATOR_VERSION,
        "constructed_6_8b": False,
    }
    path = root / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _write_ready(root: Path, *, manifest: dict[str, Any], repo: Path | None = None) -> dict[str, Any]:
    blob = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    marker = {
        "marker": "READY_V031",
        "profile": manifest["profile"],
        "manifest_sha256": sha256_text(blob),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(repo),
        "generator": "scripts/prepare_v031_dataset.py",
        "generator_version": GENERATOR_VERSION,
    }
    (root / READY_NAME).write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return marker


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_v031_dataset(
    root: str | Path,
    *,
    profile: str = "v031",
    expected_seed: int | None = None,
) -> dict[str, Any]:
    """Read-only. Does not generate, split, or repair."""
    if profile not in PROFILES:
        raise DatasetContractError([f"unknown profile {profile!r}"])
    root = Path(root)
    failures: list[str] = []
    ready_path = root / READY_NAME
    manifest_path = root / MANIFEST_NAME
    if not ready_path.is_file():
        failures.append(f"missing {READY_NAME}")
    if not manifest_path.is_file():
        failures.append(f"missing {MANIFEST_NAME}")
    inspection = inspect_pack(root, profile=profile)
    failures.extend(inspection["failures"])
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    ready = _read_json(ready_path) if ready_path.is_file() else {}
    if manifest:
        if str(manifest.get("profile")) != profile:
            failures.append(f"manifest profile={manifest.get('profile')!r} want {profile}")
        if expected_seed is not None and int(manifest.get("seed") or -1) != int(expected_seed):
            failures.append(f"seed={manifest.get('seed')} want {expected_seed}")
        if manifest.get("pwm") is not False:
            failures.append("manifest pwm must be false")
        if manifest.get("hf_data") is not False:
            failures.append("hf_data must be false")
        if manifest.get("audit") != "PASS":
            failures.append(f"manifest audit={manifest.get('audit')!r}")
        stored = manifest.get("hashes") or {}
        for rel, digest in inspection["hashes"].items():
            if stored.get(rel) != digest:
                failures.append(f"hash drift {rel}")
        blob = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        if ready and ready.get("manifest_sha256") != sha256_text(blob):
            failures.append("READY marker manifest hash mismatch")
        if ready and str(ready.get("profile")) != profile:
            failures.append(f"READY profile={ready.get('profile')!r} want {profile}")
    if failures:
        raise DatasetContractError(failures)
    return {
        "pass": True,
        "root": str(root),
        "profile": profile,
        "manifest": manifest,
        "ready": ready,
        "hashes": inspection["hashes"],
        "mutated": False,
        "constructed_6_8b": False,
    }


def prepare_v031_dataset(
    root: str | Path,
    *,
    n: int,
    seed: int = 11,
    profile: str = "v031",
    repo: Path | None = None,
) -> dict[str, Any]:
    """CPU/2080. Creates the pack. H200 must not call this."""
    if profile not in PROFILES:
        raise DatasetContractError([f"unknown profile {profile!r}"])
    if profile == "v031" and int(n) < V031_MIN_N:
        raise DatasetContractError([f"n={n} < {V031_MIN_N} for profile v031; use --profile cpu_dev"])
    if profile == "cpu_dev" and int(n) < CPU_DEV_MIN_N:
        raise DatasetContractError([f"cpu_dev n={n} < {CPU_DEV_MIN_N} (need episode_index 9 in heldout)"])
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    ready_path = root / READY_NAME
    if ready_path.exists():
        ready_path.unlink()
    repo = repo or ROOT
    config = load_simulation(repo / "configs" / "simulation" / "milestone1.yaml")
    write_curriculum(
        root,
        config,
        seed=int(seed),
        n_episodes=int(n),
        length=None if profile == "v031" else CPU_DEV_LENGTH,
        lengths=None if profile != "v031" else PHASE_LENGTHS,
    )
    write_heldout_split(root)
    audit = audit_curriculum(root, gate=False)
    (root / REPORT_NAME).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    inspection = inspect_pack(root, profile=profile)
    manifest = _write_manifest(root, profile=profile, seed=seed, inspection=inspection, audit=audit)
    if not inspection["pass"] or manifest["audit"] != "PASS":
        if ready_path.exists():
            ready_path.unlink()
        raise DatasetContractError(list(inspection["failures"]))
    marker = _write_ready(root, manifest=manifest, repo=repo)
    return {
        "pass": True,
        "root": str(root),
        "profile": profile,
        "ready": str(ready_path),
        "manifest": manifest,
        "marker": marker,
        "constructed_6_8b": False,
    }


def dataset_is_v031(training: Any) -> bool:
    name = str(getattr(training, "name", "") or "")
    root = str(getattr(training, "dataset_root", "") or "").replace("\\", "/").rstrip("/")
    return name == "mina_6_8b_v03" or root.endswith("dataset/mina_6_8b_v03") or Path(root).name == DATASET_NAME


def assert_v031_train_dataset(repo: Path, training: Any) -> dict[str, Any]:
    """Refuse v0.3.1 train without a production READY pack. Does not construct 6.8B."""
    if not dataset_is_v031(training):
        return {"required": False, "pass": True}
    raw = str(training.dataset_root)
    root = Path(raw)
    if not root.is_absolute():
        root = Path(repo) / raw
    return verify_v031_dataset(root, profile="v031")
