# RunPod — RTX PRO 6000 Blackwell first

Foundation tag exists: `MINAKANUSHI-v0.1-foundation`.

Current cash: **~$30**. That is Stage A only. November (100–200k RUB) is
Stage B / `research_v01`. See `docs/TRAINING_PLAN.md`.

## Order

```text
Stage A   RTX PRO 6000 BW    gpu_train_v01 6.2M     2–3 h
          → fill docs/GPU_BRINGUP_6000BW.md
          → STOP POD

Decision  PASS + scaling signs  → Stage B later
          FAIL                  → no medium, no H200

Stage B   6000 BW             MINA-medium 100–300M   November
Stage C   H100 / H200         research_v01 1.3B      November
```

Community Cloud ~$1.69/h (check live). $30 ≈ 14–17 h. Cap the pod.

## Stage A commands (pod)

```text
git clone --branch MINAKANUSHI-v0.1-foundation <repo>
python -m pip install -e ".[test]"
python -m pytest tests
python scripts/generate_dataset.py --root dataset --n 8 --length 12
# then train gpu_train_v01 on cuda / bf16 — fill the bring-up report
```

Do not instantiate `research_v01` on this pod. Do not download foundation
models. Dataset is SyntheticWorld v1.

## Not now

- Stage B YAML until Maga reads the report
- H100/H200 for bring-up
- overnight pod without auto-stop
- treating `loss ↓` as success
