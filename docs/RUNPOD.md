# RunPod — 6.8B train is H200, not 6000 BW

**Target profile:** `minakanushi_6_8b` (6.8B). Spec: `docs/MINA_6_8B_TRAINING.md`.  
**Gate:** `docs/GATE_6_8B_PRETRAIN.md`. Frozen at `7aba976`.

Do not `MinakanushiSystem` from `minakanushi_6_8b.yaml` on CPU or RTX PRO 6000.

**Do not terminate this H200.** Do not stop. Maga prepares the machine.

## Current iron — 1× H200 SXM

```text
image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
GPU:   1× H200 SXM  141 GB
RAM:   188 GB
vCPU:  12
disk:  2250 GB
```

GPU ~$4.59/h + disk. Leave it UP while work is in flight.

1× H200 is the sanity machine. Full AdamW 6.8B is tight on 141 GB
(weights+opt+grad ≈ 110–136 GB before activations). If OOM: do not add
slots, do not add layers. Activation checkpoint is already on. Next iron
is 2× H200 or 1× B300.

## Operator — Maga pastes this

On the laptop, after the pre-train commit is on `origin/main`:

```text
git push origin main
```

On the H200 (do not pip-install numpy):

```text
cd /workspace
git clone https://github.com/MagistrTheOne/NULLXES-MINAKANUSHI.git NULLXES-MINAKANUSHI
cd NULLXES-MINAKANUSHI
git fetch origin
git checkout main
git pull --ff-only origin main
python -m pip install --break-system-packages --no-deps -e .
python -u scripts/sanity_pretrain.py --out experiments/gate_6_8b_pretrain
python -u scripts/generate_6_8b_curriculum.py --root dataset/mina_6_8b --n 2 --length 12
```

Or: `bash scripts/pod_6_8b_pretrain.sh`

Sanity train is a **later** paste, not the first command:

```text
torchrun --nproc_per_node=1 scripts/train.py \
  --config configs/training/mina_6_8b_sanity.yaml \
  --out experiments/mina_6_8b_sanity
```

That constructs 6.8B. Only after stack JSON exists. Detach with `nohup` /
`torchrun` in background. **Do not terminate.**

Closed: Yunmu, Gate 10, FP16, 48 layers, extra slots, λ as a substitute for data.
