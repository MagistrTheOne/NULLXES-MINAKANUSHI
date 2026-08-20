# RunPod — stack instrument first, 6.8B later

**Target profile:** `minakanushi_6_8b` (6.8B). Spec: `docs/MINA_6_8B_TRAINING.md`.  
**Do not** `MinakanushiSystem` from `minakanushi_6_8b.yaml` on the 6000 BW pod.

## Next cash — Gate 03 Revision Validation

Tag: `MINAKANUSHI-revision-gate` (`0ea5062`). Profile: `gpu_train_v01` only.
Wall clock 30–60 min. Validate first. Train only if revision metrics move.
`λ_revision = 1.0`. Do not retune λ. Success is not `loss ↓`.

```text
git clone <repo>
python -m pip install -e ".[test]"
python -m pytest tests -q
python scripts/gate03_revision_validate.py \
  --training configs/training/stage_a_gpu_train_v01.yaml \
  --out experiments/gate03_revision
```

Fill After column in `docs/GATE_03_REVISION_VALIDATION.md`. STOP POD.

Closed: 6.8B, H200, humanoid, Yunmu, large datasets.

## Stage A (done)

```text
Stage A   RTX PRO 6000 BW    gpu_train_v01 6.2M
          CUDA / bf16 / AMP / *.mina
          docs/GPU_BRINGUP_6000BW.md
```

Community Cloud ~$1.69–2.40/h. Cap the pod.

## November (100–200k RUB)

2× H200 or 1× B300. FSDP, activation checkpointing, episode streaming.  
Yunmu package: `models/MINA-6.8B` + docs. Weights after curriculum gates.
Only after Gate 03 revision is a live signal.

## Not now

- training 6.8B on 96 GB
- overnight without auto-stop
- `loss ↓` as success
- LLM downloads
