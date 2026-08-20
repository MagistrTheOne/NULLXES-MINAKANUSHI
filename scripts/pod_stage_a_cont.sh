#!/usr/bin/env bash
# Continue Stage A after clone. Patch torch 2.8 checkpoint load. Never construct 6.8B.
set -euo pipefail
export PYTHONUNBUFFERED=1
REPO=/workspace/NULLXES-MINAKANUSHI
OUT="${REPO}/experiments/stage_a"
cd "${REPO}"
mkdir -p "${OUT}"

python - <<'PY'
from pathlib import Path
p = Path("minakanushi/training/checkpoint.py")
t = p.read_text()
old = 'payload = torch.load(io.BytesIO(zf.read(WEIGHTS_NAME)), map_location="cpu")'
new = 'payload = torch.load(io.BytesIO(zf.read(WEIGHTS_NAME)), map_location="cpu", weights_only=False)'
if "weights_only=False" not in t:
    if old not in t:
        raise SystemExit("checkpoint.py load site not found")
    p.write_text(t.replace(old, new, 1))
    print("patched torch.load weights_only=False")
else:
    print("already patched")
PY

echo "=== PYTEST ==="
python -m pytest tests -q | tee "${OUT}/pytest.txt"

echo "=== DATASET ==="
python scripts/generate_dataset.py --root dataset --n 8 --length 12 | tee "${OUT}/dataset.txt"

cat > configs/training/stage_a_gpu_train_v01.yaml <<'YAML'
stage: 0
name: stage_a_gpu_train_v01
architecture: configs/architecture/gpu_train_v01.yaml
simulation: configs/simulation/milestone1.yaml
dataset_name: stage0_synthetic
n_overfit_episodes: 16
seed: 7
steps: 200
batch_size: 1
sequence_length: 12
learning_rate: 0.001
weight_decay: 0.0001
grad_clip: 1.0
log_every: 10
eval_every: 50
checkpoint_every: 100
precision: bf16
device: cuda
lambdas:
  state: 1.0
  temporal: 1.0
  future: 0.5
  uncertainty: 0.3
  causal: 0.2
  memory: 0.4
  action: 0.3
  representation: 0.05
  belief: 0.5
regularizer:
  isotropic_weight: 0.05
  counterfactual_margin: 0.25
YAML

echo "=== PROBE gpu_train_v01 ==="
python - <<'PY'
import json, time
from pathlib import Path
import torch
from minakanushi.architecture.config import load_architecture
from minakanushi.architecture.model import MinakanushiSystem
from minakanushi.training.parameter_inventory import estimate_parameters
from minakanushi.training.checkpoint import save_mina, load_mina

root = Path("/workspace/NULLXES-MINAKANUSHI")
cfg = load_architecture(root / "configs/architecture/gpu_train_v01.yaml")
assert cfg.latent_dim == 256 and cfg.core_depth == 6
est = estimate_parameters(cfg)["total_estimate"]
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
sys = MinakanushiSystem(cfg).cuda()
construct_s = time.perf_counter() - t0
n = sum(p.numel() for p in sys.parameters())
alloc = torch.cuda.memory_allocated() / 1024**2
reserved = torch.cuda.memory_reserved() / 1024**2
loss = sum(p.float().square().mean() for p in sys.parameters())
loss.backward()
finite = all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in sys.parameters())
out = root / "experiments/stage_a"
save_mina(out / "probe.mina", sys, extras={"stage": "A-probe"})
fresh = MinakanushiSystem(cfg)
manifest = load_mina(out / "probe.mina", fresh)
report = {
    "profile": "gpu_train_v01",
    "params_numel": int(n),
    "params_estimate": int(est),
    "construct_s": round(construct_s, 3),
    "vram_allocated_mb": round(alloc, 1),
    "vram_reserved_mb": round(reserved, 1),
    "grad_finite": bool(finite),
    "checkpoint_identity": manifest["architecture"] == "MINAKANUSHI",
    "bf16_supported": bool(torch.cuda.is_bf16_supported()),
}
print(json.dumps(report, indent=2))
(out / "probe.json").write_text(json.dumps(report, indent=2))
PY

echo "=== TRAIN gpu_train_v01 cuda bf16 ==="
python scripts/train.py --config configs/training/stage_a_gpu_train_v01.yaml --out experiments/stage_a | tee "${OUT}/train.log"
echo "=== STAGE A DONE $(date -Is) ==="
ls -lh experiments/stage_a
nvidia-smi
