# RunPod — stack instrument first, 6.8B later

**Target profile:** `minakanushi_6_8b` (6.8B). Spec: `docs/MINA_6_8B_TRAINING.md`.  
**This cash (~$30):** 1× RTX PRO 6000 BW, **`gpu_train_v01` 6.2M only**.

Do not `MinakanushiSystem` from `minakanushi_6_8b.yaml` on this pod.

## Today

```text
Stage A   RTX PRO 6000 BW    gpu_train_v01 6.2M     2–3 h
          CUDA / bf16 / AMP / dataloader / *.mina
          fill docs/GPU_BRINGUP_6000BW.md
          STOP POD
```

Community Cloud ~$1.69/h. Cap the pod.

```text
git clone --branch MINAKANUSHI-v0.1-foundation <repo>
python -m pip install -e ".[test]"
python -m pytest tests
python scripts/generate_dataset.py --root dataset --n 8 --length 12
# train gpu_train_v01 cuda bf16 — not 6.8B
```

## November (100–200k RUB)

2× H200 or 1× B300. FSDP, activation checkpointing, episode streaming.  
Yunmu package: `models/MINA-6.8B` + docs. Weights after curriculum gates.

## Not now

- training 6.8B on 96 GB
- overnight without auto-stop
- `loss ↓` as success
- LLM downloads
