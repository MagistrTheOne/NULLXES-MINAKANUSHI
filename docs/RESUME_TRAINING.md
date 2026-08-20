# Resume training v0.2

```text
ARCHITECTURE FREEZE — resume restores state. It does not change architecture.
```

Weights-only load is a **clone**. Production resume is the **same model**:

```text
step64
  + optimizer
  + RNG (python / numpy / torch)
  + dataset cursor
  + scheduler
  + identity extras
  → step65+
```

```text
python scripts/train.py --config configs/training/mina_6_8b_v02.yaml \
  --out experiments/mina_6_8b_v02 \
  --resume MINA-6.8B-IdentityBound.mina
```

CPU replay gate: save at step 10, load, step 11 must match a continuous run's step 11 (`tests/unit/test_resume_replay.py`). CUDA bitwise equality is not the gate.
