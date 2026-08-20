# MINA training plan — Maga sign-off

**Target:** NULLXES MINAKANUSHI **6.8B** (`minakanushi_6_8b` / `models/MINA-6.8B`).  
Architecture frozen at `7aba976`. Gate: `docs/GATE_6_8B_PRETRAIN.md`.

`gpu_train_v01` 6.2M was the stack instrument. Gate 03B closed it.

---

## Order

```text
7aba976
  ↓
Freeze MINA foundation
  ↓
Prepare 6.8B training stack (this gate)
  ↓
Episode curriculum generation
  ↓
H200 / B300 sanity pretrain
  ↓
MINA-6.8B checkpoint
  ↓
only then Yunmu humanoid
```

Do not construct 6.8B on CPU or RTX PRO 6000.
Do not add layers, slots, or modules.
bf16 only. FSDP2 / ZeRO-3. Sharded `*.mina` must resume.

Success is not `loss ↓`. Belief revision, memory effect, action causality,
uncertainty calibration, causal consistency.

---

Maga: 6.8B is the training contract. Scale this MINA.
