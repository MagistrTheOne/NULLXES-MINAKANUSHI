"""Move *.mina to HF repo root and publish the model card. Token from HF_TOKEN."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
    HfApi,
    login,
)

REPO_ID = "MagistrTheOne/MINAKANUSHI-6.8B"
README = Path("/workspace/HF_README.md")


def main() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token.startswith("hf_"):
        token_path = Path("/workspace/.hf_token")
        if token_path.exists():
            token = token_path.read_text(encoding="utf-8").strip()
    if not token.startswith("hf_"):
        raise SystemExit("HF_TOKEN missing")
    login(token=token, add_to_git_credential=False)
    api = HfApi(token=token)

    files = set(api.list_repo_files(REPO_ID))
    ops: list = []

    copies = [
        ("checkpoints/minakanushi_stage0_step64.mina", "minakanushi_stage0_step64.mina"),
        ("checkpoints/v0.2/minakanushi_stage0_step128.mina", "minakanushi_stage0_step128.mina"),
        ("checkpoints/v0.2/metrics.jsonl", "metrics_v02.jsonl"),
        ("checkpoints/v0.2/train.log", "train_v02.log"),
    ]
    for src, dst in copies:
        if src not in files:
            raise SystemExit(f"missing source {src}")
        if dst not in files:
            ops.append(CommitOperationCopy(src_path_in_repo=src, path_in_repo=dst))

    if not README.exists():
        raise SystemExit(f"missing {README}")
    ops.append(
        CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=README.read_bytes(),
        )
    )

    deletes = [
        "checkpoints/minakanushi_stage0_step64.mina",
        "checkpoints/v0.2/minakanushi_stage0_step128.mina",
        "checkpoints/v0.2/metrics.jsonl",
        "checkpoints/v0.2/train.log",
        "checkpoints/v0.2/README.md",
    ]
    for path in deletes:
        if path in files:
            ops.append(CommitOperationDelete(path_in_repo=path))

    info = api.create_commit(
        repo_id=REPO_ID,
        repo_type="model",
        operations=ops,
        commit_message="Move checkpoints to repo root and publish 6.8B parameter card",
    )
    print("commit", info.commit_url)
    print("files", sorted(api.list_repo_files(REPO_ID)))


if __name__ == "__main__":
    main()
