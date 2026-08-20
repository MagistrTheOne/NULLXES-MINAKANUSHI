"""Upload Status Core artifacts to Hugging Face. Token from HF_TOKEN only."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, login

REPO_ID = "MagistrTheOne/MINAKANUSHI-6.8B"
STAGING = Path("/workspace/hf_MINAKANUSHI-6.8B")
CKPT = Path("/workspace/NULLXES-MINAKANUSHI/experiments/mina_6_8b_status_core_researched")


def main() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token.startswith("hf_"):
        raise SystemExit("HF_TOKEN missing")
    login(token=token, add_to_git_credential=False)
    api = HfApi(token=token)

    readme = STAGING / "HF_README.md"
    if readme.exists():
        readme.replace(STAGING / "README.md")
    train_yaml = STAGING / "mina_6_8b_status_core_researched.yaml"
    if train_yaml.exists():
        train_yaml.replace(STAGING / "configs" / train_yaml.name)

    dest_ckpt = STAGING / "checkpoints" / "minakanushi_stage0_step64.mina"
    src_ckpt = CKPT / "minakanushi_stage0_step64.mina"
    if src_ckpt.exists() and not dest_ckpt.exists():
        dest_ckpt.hardlink_to(src_ckpt)

    for name in ("metrics.jsonl", "dataset_audit.json"):
        src = CKPT / name
        if src.exists():
            shutil.copy2(src, STAGING / name)

    api.create_repo(REPO_ID, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(STAGING),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Publish NULLXES MINAKANUSHI 6.8B Status Core (Researched)",
        ignore_patterns=["*.tmp", "__pycache__"],
    )
    print(f"published https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
