#!/usr/bin/env bash
# MINAKANUSHI 6.8B PRE-TRAIN onboard — 1× H200 SXM.
# Image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
# Stack check + curriculum. Sanity train is a later paste.
# Do NOT terminate. Do NOT stop. Do NOT pip-install numpy.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export GIT_TERMINAL_PROMPT=0

WS=/workspace
REPO="${WS}/NULLXES-MINAKANUSHI"
OUT="${REPO}/experiments/gate_6_8b_pretrain"
SHA="${SHA:-}"

echo "=== 6.8B PRE-TRAIN STACK $(date -Is) ==="
nvidia-smi
python - <<'PY'
import json, torch
assert torch.cuda.is_available(), "cuda required"
p = torch.cuda.get_device_properties(0)
name = torch.cuda.get_device_name(0)
print(json.dumps({
    "torch": torch.__version__,
    "numpy": __import__("numpy").__version__,
    "name": name,
    "bf16": bool(torch.cuda.is_bf16_supported()),
    "vram_gb": round(p.total_memory / 1024**3, 2),
}, indent=2))
if "H200" not in name:
    raise SystemExit(f"this onboard expects H200, got {name}")
PY

if [ ! -d "${REPO}/.git" ]; then
  git clone https://github.com/MagistrTheOne/NULLXES-MINAKANUSHI.git "${REPO}"
fi
cd "${REPO}"
git fetch origin
if [ -n "${SHA}" ]; then
  git checkout "${SHA}"
  test "$(git rev-parse --short HEAD)" = "${SHA}" || test "$(git rev-parse HEAD)" = "${SHA}"
else
  git checkout main
  git pull --ff-only origin main
fi
git log -1 --oneline

python -m pip install --break-system-packages --no-deps -e .

mkdir -p "${OUT}"
echo "=== sanity_pretrain (does not construct 6.8B) ==="
python -u scripts/sanity_pretrain.py --out "${OUT}" | tee "${OUT}/sanity_stack.log"

echo "=== episode curriculum n=2 ==="
python -u scripts/generate_6_8b_curriculum.py \
  --root "${REPO}/dataset/mina_6_8b" \
  --n 2 \
  --length 12 | tee "${OUT}/curriculum.log"

echo "=== STACK READY $(date -Is) ==="
echo "Pod stays UP. Do not terminate. Do not stop."
echo "Sanity train (later, 20 steps, 1× H200, may be tight on 141 GB):"
echo "  torchrun --nproc_per_node=1 scripts/train.py --config configs/training/mina_6_8b_sanity.yaml --out experiments/mina_6_8b_sanity"
