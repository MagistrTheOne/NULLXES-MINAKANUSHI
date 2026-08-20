#!/usr/bin/env bash
# Gate 03B diagnostic on RTX PRO 6000 BW. gpu_train_v01 only.
# Eval, not a full train. Never construct 6.8B.
# Do NOT terminate the pod. Community Cloud stop/terminate wipes the machine.
# Do NOT pip-install numpy — the torch image already has a matching build.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export GIT_TERMINAL_PROMPT=0

WS=/workspace
REPO="${WS}/NULLXES-MINAKANUSHI"
OUT="${REPO}/experiments/gate03b"
SHA=7fd8ef6

echo "=== GATE 03B START $(date -Is) ==="
nvidia-smi
python - <<'PY'
import json, torch
assert torch.cuda.is_available(), "cuda required"
p = torch.cuda.get_device_properties(0)
print(json.dumps({
    "torch": torch.__version__,
    "numpy": __import__("numpy").__version__,
    "name": torch.cuda.get_device_name(0),
    "bf16": bool(torch.cuda.is_bf16_supported()),
    "vram_gb": round(p.total_memory / 1024**3, 2),
}, indent=2))
name = torch.cuda.get_device_name(0)
if "H200" in name:
    raise SystemExit(f"wrong GPU for Gate 03B: {name}")
PY

if [ ! -d "${REPO}/.git" ]; then
  git clone https://github.com/MagistrTheOne/NULLXES-MINAKANUSHI.git "${REPO}"
fi
cd "${REPO}"
git fetch origin
git checkout "${SHA}"
test "$(git rev-parse --short HEAD)" = "${SHA}"

python -m pip install --break-system-packages --no-deps -e .

mkdir -p "${OUT}"
echo "=== GATE 03B n=1000 gpu_train_v01 eval, nohup ==="
nohup python -u scripts/gate03b_hidden_direction.py \
  --training configs/training/stage_a_gpu_train_v01.yaml \
  --n 1000 \
  --out "${OUT}" > "${OUT}/gate03b.log" 2>&1 &
echo "pid=$!"
echo "tail -f ${OUT}/gate03b.log"
echo "Pod stays UP until JSON exists. Do not terminate."
