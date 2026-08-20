# MINA training plan — Maga sign-off

NULLXES MINAKANUSHI. Not an LLM scale-up. Weights grow only after the
internal world gets more accurate.

**Now:** tag `MINAKANUSHI-v0.1-foundation` exists. Budget **~$30**.  
**November:** **100–200k RUB** for medium + `research_v01` on H100/H200.  
**Not now:** multi-GPU 5–10B, vision, language, embodiment.

---

## Order

```text
COMMIT + TAG          done
        ↓
RTX PRO 6000 BW       ~$1.69/h Community
        ↓
Stage A  gpu_train_v01  6.2M   2–3 hours
        ↓
docs/GPU_BRINGUP_6000BW.md     mandatory artifact
        ↓
Decision
   FAIL  → stop, no medium
   PASS + scaling signs → Stage B when November money lands
        ↓
MINA-medium  ~100–300M  (latent 1024, slots 256/512, depth 12)
        ↓
H100/H200
        ↓
research_v01  1.3B
        ↓
only then multi-GPU 5–10B+
```

Stage B is **not** automatic after A. NaN, VRAM leak, or dataset bottleneck
→ medium is forbidden.

---

## What 6.2M is

Not “LLM with 6 million parameters.”

It already contains: world slots, memory, uncertainty, future branches,
authority, runtime, belief. Stage A is a **causality-loop** test on Blackwell.

---

## What success is

Not `loss ↓`.

```text
Belief     correct state ↑
Memory     with-memory > without-memory
Action     WAIT future ≠ MOVE_TO future
OOD        new combination, not train replay
```

If bigger model only makes step/s worse and belief stays flat, stop scaling.

---

## Money

| When | Money | Allowed |
|---|---|---|
| now | ~$30 ≈ 14–17 h on 6000 BW | Stage A + report; maybe a second A if first dies |
| November | 100–200k RUB | Stage B smoke + `research_v01` on H100/H200 |
| later | more | 5–10B only after 1.3B proved |

Do not spend November money on Stage B if the bring-up report says FAIL.

---

## Profiles (same architecture family)

| Name | Params | When |
|---|---:|---|
| `cpu_dev` | 0.21M | tests, already done |
| `gpu_train_v01` | 6.2M | Stage A now |
| MINA-medium | ~100–300M | after A PASS, November |
| `research_v01` | 1.3B | H100/H200 after medium |
| 5–10B | later | multi-GPU, not this year unless Maga opens it |

Learned: perception, NPF, DWC, uncertainty, memory read, future.  
Not learned: slots, constraints, authority, ActionIntent.

---

Maga: this file is the training contract until revised.
