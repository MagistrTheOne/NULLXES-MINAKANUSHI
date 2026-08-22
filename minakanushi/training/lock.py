"""Write the v0.3.1 pre-training baseline pack. Never constructs 6.8B."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from minakanushi.training.baseline import inspect_mina, sha256_file
from minakanushi.training.capability import cpu_trainer, packet_snapshot

ROOT = Path(__file__).resolve().parents[2]

HARDWARE = {
    "phase": "v0.3.1 Phase 1",
    "gpu": "H200",
    "parallelism": "fsdp2_zero3",
    "precision": "bf16",
    "canonical_checkpoint": "*.mina",
    "steps": 1000,
    "stop": "stop after 1000 steps; do not train forever",
    "forbidden": [
        "construct minakanushi_6_8b on CPU",
        "construct minakanushi_6_8b on RTX PRO 6000",
        "train 6.8B on 1x H100 80GB",
        "DWC / latent / slots / new heads / language adapter / RGB",
    ],
}


LOCAL_DIRTY_PREFIXES = ("dataset/", "artifacts/", "experiments/", "outputs/", "logs/")


def classify_git_status(repo: Path | None = None) -> dict[str, Any]:
    """Split local data dirt from code dirt. DIRTY on git_commit.txt is not enough."""
    repo = repo or ROOT
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "UNKNOWN", "code_dirty": False, "local_dirty": [], "code_dirty_files": []}
    local_dirty: list[str] = []
    code_dirty_files: list[str] = []
    for line in porcelain.splitlines():
        path = line[3:].strip().replace("\\", "/").split(" -> ")[-1]
        if any(path.startswith(prefix) for prefix in LOCAL_DIRTY_PREFIXES):
            local_dirty.append(path)
        else:
            code_dirty_files.append(path)
    return {
        "git_commit": head,
        "code_dirty": bool(code_dirty_files),
        "local_dirty": local_dirty,
        "code_dirty_files": code_dirty_files,
        "dirty_files": local_dirty + code_dirty_files,
    }


def git_commit(repo: Path | None = None) -> str:
    status = classify_git_status(repo)
    head = str(status.get("git_commit") or "UNKNOWN")
    if status.get("code_dirty"):
        return f"{head} CODE_DIRTY"
    if status.get("local_dirty"):
        return f"{head} local-only"
    return head


def write_run_manifest(
    dest: str | Path,
    *,
    training_config: dict[str, Any] | None = None,
    checkpoint_sha256: str = "MISSING",
    git_status: dict[str, Any] | None = None,
    n_episodes: int = 1000,
    n_train: int = 900,
    n_heldout: int = 100,
) -> dict[str, Any]:
    train = training_config or {}
    status = git_status or classify_git_status()
    manifest = {
        "model": "MINAKANUSHI-6.8B",
        "architecture": "frozen",
        "architecture_lock": "7aba976",
        "base_checkpoint": "step128",
        "checkpoint_sha256": checkpoint_sha256,
        "dataset": "mina_6_8b_v03",
        "episodes": int(n_episodes),
        "train_split": int(n_train),
        "heldout_split": int(n_heldout),
        "hardware": "H200",
        "steps": int(train.get("steps") or 1000),
        "eval_every": int(train.get("eval_every") or 50),
        "stop_condition": "manual after 1000 steps",
        "rgb": False,
        "pwm": False,
        "git_commit": status.get("git_commit"),
        "code_dirty": bool(status.get("code_dirty")),
        "compare": "capability_before vs capability_after, not step128 loss vs step1128 loss",
    }
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def write_reference_inference(out_dir: Path) -> Path:
    """cpu_dev snapshot only. Research-scale inference is load_mina on H200."""
    import torch

    trainer = cpu_trainer(12)
    pkt = trainer.unroll(1, scenario="const_velocity", episode_index=0, seed=7, length=12)
    path = out_dir / "reference_inference.pt"
    torch.save(packet_snapshot(pkt), path)
    return path


def lock_baseline(
    out_dir: str | Path,
    *,
    mina: str | Path | None = None,
    dataset_root: str | Path | None = None,
    training_config: str | Path | None = None,
    capability_path: str | Path | None = None,
    run_capability: bool = False,
    write_inference: bool = True,
    require_mina: bool = False,
    repo: Path | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo = repo or ROOT
    report: dict[str, Any] = {"out_dir": str(out_dir), "constructed_6_8b": False}

    status = classify_git_status(repo)
    commit = git_commit(repo)
    (out_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    (out_dir / "git_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["git_commit"] = commit
    report["git_status"] = status

    cfg_src = Path(training_config) if training_config else repo / "configs" / "training" / "mina_6_8b_v03.yaml"
    dest_yaml = out_dir / "training_config.yaml"
    shutil.copyfile(cfg_src, dest_yaml)
    train_raw = yaml.safe_load(cfg_src.read_text(encoding="utf-8"))
    hardware = dict(HARDWARE)
    hardware["training_name"] = train_raw.get("name")
    hardware["dataset_root"] = train_raw.get("dataset_root")
    hardware["dataset_split"] = train_raw.get("dataset_split")
    hardware["yaml_steps"] = train_raw.get("steps")
    hardware["device"] = train_raw.get("device")
    (out_dir / "hardware.json").write_text(json.dumps(hardware, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["hardware"] = hardware

    mina_path = Path(mina) if mina else None
    sha_path = out_dir / "checkpoint.sha256"
    if mina_path is not None and mina_path.is_file():
        inventory = inspect_mina(mina_path)
        sha_path.write_text(inventory["sha256"] + "\n", encoding="utf-8")
        (out_dir / "metrics_before.json").write_text(
            json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report["checkpoint_sha256"] = inventory["sha256"]
        report["research_scale"] = bool(inventory["research_scale"])
        if inventory["research_scale"]:
            (out_dir / "REFERENCE_INFERENCE.txt").write_text(
                "research-scale: do not construct 6.8B here. reference_inference.pt is cpu_dev.\n",
                encoding="utf-8",
            )
    else:
        sha_path.write_text("MISSING\n", encoding="utf-8")
        (out_dir / "metrics_before.json").write_text(
            json.dumps(
                {
                    "status": "MISSING",
                    "reason": "no *.mina on this machine",
                    "constructed_6_8b": False,
                    "next": "run lock on the box that holds step128.mina, or pass --mina",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        report["checkpoint_sha256"] = "MISSING"
        report["research_scale"] = None
        if require_mina:
            raise FileNotFoundError("step128 *.mina is required for a closed baseline")

    data_root = Path(dataset_root) if dataset_root else repo / "dataset" / "mina_6_8b_v03"
    from minakanushi.training.curriculum_audit import audit_curriculum

    if data_root.exists():
        dataset_report = audit_curriculum(data_root, gate=False)
        (out_dir / "dataset_report.json").write_text(
            json.dumps(dataset_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report["dataset_report"] = {"n_episodes": dataset_report["n_episodes"], "pwm": dataset_report["pwm"]}
    else:
        (out_dir / "dataset_report.json").write_text(
            json.dumps({"status": "MISSING", "dataset_root": str(data_root)}, indent=2) + "\n",
            encoding="utf-8",
        )
        report["dataset_report"] = {"status": "MISSING"}

    cap_dest = out_dir / "capability_before.json"
    cap_src = Path(capability_path) if capability_path else repo / "artifacts" / "v031" / "capability" / "capability_report.json"
    if run_capability:
        from minakanushi.training.capability import run_capability_suite

        suite = run_capability_suite(out_dir / "capability_run")
        cap_dest.write_text(json.dumps(suite, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        report["capability"] = "ran_cpu_dev"
    elif cap_src.is_file():
        shutil.copyfile(cap_src, cap_dest)
        report["capability"] = str(cap_src)
    else:
        cap_dest.write_text(
            json.dumps(
                {
                    "status": "NOT_RUN",
                    "protocol": "python scripts/gate_capability.py --out artifacts/v031/capability",
                    "ledger": "docs/MINA_CAPABILITY_LEDGER.md",
                    "note": "do not update proven=PASS from this stub",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        report["capability"] = "NOT_RUN"

    if write_inference:
        inf = write_reference_inference(out_dir)
        report["reference_inference"] = str(inf)
    else:
        (out_dir / "reference_inference.MISSING.txt").write_text(
            "skipped. cpu_dev snapshot is scripts/lock_v031_baseline.py default.\n",
            encoding="utf-8",
        )
        report["reference_inference"] = "SKIPPED"

    n_episodes = 0
    n_train = 0
    n_heldout = 0
    ds = report.get("dataset_report") or {}
    if isinstance(ds, dict) and "n_episodes" in ds:
        n_episodes = int(ds["n_episodes"])
    splits = data_root / "splits.json"
    if splits.is_file():
        split_info = json.loads(splits.read_text(encoding="utf-8"))
        n_train = int(split_info.get("n_train") or 0)
        n_heldout = int(split_info.get("n_heldout") or 0)
        n_episodes = int(split_info.get("n_episodes") or n_episodes)
    write_run_manifest(
        out_dir / "run_manifest.json",
        training_config=train_raw,
        checkpoint_sha256=str(report.get("checkpoint_sha256") or "MISSING"),
        git_status=status,
        n_episodes=n_episodes or 1000,
        n_train=n_train or 900,
        n_heldout=n_heldout or 100,
    )
    report["run_manifest"] = str(out_dir / "run_manifest.json")

    (out_dir / "lock_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
