# RunPod — stack instrument first, 6.8B later

**Target profile:** `minakanushi_6_8b` (6.8B). Spec: `docs/MINA_6_8B_TRAINING.md`.  
**Do not** `MinakanushiSystem` from `minakanushi_6_8b.yaml` on the 6000 BW pod.

## Next cash — Gate 03B hidden direction

Parent exam: Gate 03 on `main` (`3d8012e`). Profile: `gpu_train_v01` only.
Wall clock 30–60 min. **Eval diagnostic, not a full train.** `λ_revision = 1.0`.
Do not retune λ. Do not scale. Question: does hidden direction stay 0.5
across N episodes, or was that `cpu_dev` / one seed?

```text
git clone <repo>
python -m pip install -e ".[test]"
python -m pytest tests -q
python scripts/gate03b_hidden_direction.py \
  --training configs/training/stage_a_gpu_train_v01.yaml \
  --n 1000 \
  --out experiments/gate03b
```

Fill `docs/GATE_03B_HIDDEN_DIRECTION_REPORT.md` from the JSON.
Do **not** terminate the Community Cloud pod mid-run (the machine is gone).
Detach with `nohup`. Terminate only after the JSON is copied.

Closed: 6.8B, H200, humanoid, Yunmu, architecture edit, λ change.

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
