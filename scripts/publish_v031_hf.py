"""Publish v0.3.1 research checkpoint + dataset pack + ASI collection.

Does not construct 6.8B. Token from HF_TOKEN only. Run on H200 after reload PASS.

    HF_HUB_ENABLE_HF_TRANSFER=1 python scripts/publish_v031_hf.py
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, login

ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = "MagistrTheOne/MINAKANUSHI-6.8B"
DATASET_V03 = "MagistrTheOne/mina-6.8b-v03"
DATASET_V02 = "MagistrTheOne/mina-6.8b-v02"
COLLECTION_TITLE = "ASI (WorldModel)"
STEP_DIR = ROOT / "artifacts" / "v031" / "step1128"
MIRROR = STEP_DIR / "MINAKANUSHI-6.8B"
CANONICAL = "minakanushi_stage0_step1128.mina"
STAGING = Path("/workspace/hf_v031_publish")
DATASET_STAGING = Path("/workspace/hf_dataset_v03")


def _token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token.startswith("hf_"):
        raise SystemExit("HF_TOKEN missing")
    return token


def _hardlink_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    try:
        dest.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dest)


def stage_model() -> Path:
    if not (STEP_DIR / CANONICAL).is_file():
        raise SystemExit(f"missing canonical {STEP_DIR / CANONICAL}")
    if not (MIRROR / "model.safetensors.index.json").is_file():
        raise SystemExit(f"missing safetensors mirror under {MIRROR}")
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)
    readme = ROOT / "models" / "MINA-6.8B" / "HF_README.md"
    if readme.is_file():
        shutil.copy2(readme, STAGING / "README.md")
    license_src = ROOT / "LICENSE"
    if license_src.is_file():
        shutil.copy2(license_src, STAGING / "LICENSE")
    for name in (
        "config.json",
        "minakanushi_config.json",
        "MINAKANUSHI_CARD.json",
        "minakanushi_runtime.json",
        "model.safetensors.index.json",
    ):
        src = MIRROR / name
        if src.is_file():
            shutil.copy2(src, STAGING / name)
    gen_src = MIRROR / "generation" / "NO"
    if gen_src.is_file():
        dest = STAGING / "generation" / "NO"
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(gen_src, dest)
    for shard in sorted(MIRROR.glob("model-*.safetensors")):
        _hardlink_or_copy(shard, STAGING / shard.name)
    _hardlink_or_copy(STEP_DIR / CANONICAL, STAGING / CANONICAL)
    mapping = {
        "metrics.jsonl": "metrics_v031.jsonl",
        "experiment.jsonl": "experiment_v031.jsonl",
        "reference_inference.pt": "reference_inference_v031.pt",
        "reference_meta.pt": "reference_meta_v031.pt",
    }
    for src_name, dest_name in mapping.items():
        src = STEP_DIR / src_name
        if src.is_file():
            shutil.copy2(src, STAGING / dest_name)
    _stamp_research_cards(STAGING)
    return STAGING


def _stamp_research_cards(root: Path) -> None:
    extra = {
        "status": "research_checkpoint",
        "training_cycle": "v0.3.1",
        "capability_verdict": "pending_compare_v031",
        "accepted": False,
        "canonical_checkpoint": CANONICAL,
        "canonical_format": "mina",
        "pwm": False,
        "not_a_language_model": True,
    }
    for name in ("config.json", "minakanushi_config.json", "MINAKANUSHI_CARD.json"):
        path = root / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(extra)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def stage_dataset(src: Path, dest: Path) -> Path | None:
    if not src.is_dir():
        return None
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__", "*.tmp"))
    readme = dest / "README.md"
    if not readme.is_file():
        readme.write_text(
            "\n".join(
                [
                    "---",
                    "license: other",
                    "license_name: nullxes-research-license",
                    "tags:",
                    "  - minakanushi",
                    "  - world-model",
                    "  - synthetic",
                    "  - not-a-llm",
                    "---",
                    "",
                    "# mina-6.8b-v03",
                    "",
                    "NULLXES SyntheticWorld pack for MINAKANUSHI v0.3.1.",
                    "Source of truth: local generator, not a Hub video dump.",
                    "",
                    "```text",
                    "episodes: 1000",
                    "train / heldout: 900 / 100",
                    "physics / agency: 32 frames",
                    "causality / embodiment: 64 frames",
                    "pwm: false",
                    "```",
                    "",
                    "Canonical model runtime stays `*.mina`. This dataset is JSON episodes.",
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    return dest


def ensure_collection(api: HfApi, *, namespace: str) -> str:
    existing = list(api.list_collections(owner=namespace))
    for item in existing:
        title = getattr(item, "title", "") or ""
        if title == COLLECTION_TITLE or ("ASI" in title and "WorldModel" in title):
            return item.slug
    created = api.create_collection(
        title=COLLECTION_TITLE,
        namespace=namespace,
        description="NULLXES MINAKANUSHI world models. Not LLMs. Canonical: *.mina.",
        private=False,
        exists_ok=True,
    )
    return created.slug


def add_item(api: HfApi, slug: str, item_id: str, item_type: str) -> None:
    try:
        api.add_collection_item(slug, item_id=item_id, item_type=item_type, exists_ok=True)
    except TypeError:
        api.add_collection_item(slug, item_id=item_id, item_type=item_type)


def main() -> None:
    token = _token()
    login(token=token, add_to_git_credential=False)
    api = HfApi(token=token)
    model_dir = stage_model()
    api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=MODEL_REPO,
        repo_type="model",
        commit_message="v0.3.1 research checkpoint step1128 (not accepted). mina + safetensors mirror.",
        ignore_patterns=["*.tmp", "__pycache__"],
    )
    print(f"model https://huggingface.co/{MODEL_REPO}")

    v03 = ROOT / "dataset" / "mina_6_8b_v03"
    staged_v03 = stage_dataset(v03, DATASET_STAGING)
    if staged_v03 is not None:
        api.create_repo(DATASET_V03, repo_type="dataset", exist_ok=True, private=False)
        api.upload_folder(
            folder_path=str(staged_v03),
            repo_id=DATASET_V03,
            repo_type="dataset",
            commit_message="v0.3.1 SyntheticWorld pack (READY_V031). 900/100. pwm=false.",
        )
        print(f"dataset https://huggingface.co/datasets/{DATASET_V03}")

    v02 = ROOT / "dataset" / "mina_6_8b"
    if v02.is_dir() and (v02 / "index.jsonl").is_file():
        staged_v02 = stage_dataset(v02, Path("/workspace/hf_dataset_v02"))
        if staged_v02 is not None:
            api.create_repo(DATASET_V02, repo_type="dataset", exist_ok=True, private=False)
            api.upload_folder(
                folder_path=str(staged_v02),
                repo_id=DATASET_V02,
                repo_type="dataset",
                commit_message="v0.2 SyntheticWorld pack used for step128.",
            )
            print(f"dataset https://huggingface.co/datasets/{DATASET_V02}")

    slug = ensure_collection(api, namespace="MagistrTheOne")
    add_item(api, slug, MODEL_REPO, "model")
    if staged_v03 is not None:
        add_item(api, slug, DATASET_V03, "dataset")
    if v02.is_dir() and (v02 / "index.jsonl").is_file():
        add_item(api, slug, DATASET_V02, "dataset")
    print(f"collection https://huggingface.co/collections/{slug}")


if __name__ == "__main__":
    main()
