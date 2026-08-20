# MINA training plan — Maga sign-off

**Target:** NULLXES MINAKANUSHI **6.8B** (`minakanushi_6_8b` / `models/MINA-6.8B`).  
`gpu_train_v01` 6.2M is a stack **instrument**, not a product line.

**Now:** ~$30 on RTX PRO 6000 BW → prove CUDA/bf16/checkpoint on 6.2M.  
**November:** 100–200k RUB → 6.8B train on H200/B300.  
Warmcore may jointly finetune on NULLXES episodes after the pilot package.

See `docs/MINA_6_8B_TRAINING.md`.

---

## Order

```text
MINAKANUSHI 6.8B          contract (config exists, weights do not)

today  1× RTX PRO 6000 BW
  gpu_train_v01 6.2M instrument
  docs/GPU_BRINGUP_6000BW.md
  STOP — do not construct 6.8B on this pod

November
  FSDP + bf16 + activation checkpoint + Adam shard
  2× H200 or 1× B300
  physical episode curriculum
  validation: belief / memory / WAIT≠MOVE / OOD

then
  humanoid sim adapter (ActionIntent, 170–180 cm SelfModel)
  Yunmu review
  optional Warmcore joint *.mina
```

---

## Success

Not `loss ↓`. Belief, memory effect, action causality, OOD.  
If 6.8B is not better than the 6.2M instrument on those, stop.

---

Maga: 6.8B is the training contract. 6.2M is not a direction.
