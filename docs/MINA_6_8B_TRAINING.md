# MINAKANUSHI 6.8B Training Specification

**Profile:** `minakanushi_6_8b` / `models/MINA-6.8B`  
**Identity:** NULLXES · MINAKANUSHI · MINA  
**Params (formula inventory):** 6 799 130 646  
**Status:** contract. Weights do not exist yet.

This is the Yunmu / Warmcore research profile. `gpu_train_v01` (6.2M) is the
bring-up **instrument**, not a product line. Do not shrink 6.8B to fit a $30
pod. Prove the stack on 6.2M, train 6.8B on H200/B300.

```text
MINAKANUSHI 6.8B
    ↓
training curriculum (physical episodes)
    ↓
validation gates (belief / memory / causality / OOD)
    ↓
humanoid simulation (ActionIntent)
    ↓
Yunmu review + optional Warmcore joint finetune
```

6.8B parameters do not create intelligence. Diversity of **physical
cause → effect** does.

---

## 1. Hardware topology

Train 6.8B. Do not train it on RTX PRO 6000 Blackwell (96 GB is infer/demo).

| Role | Topology | Why |
|---|---|---|
| Stage A bring-up (today, ~$30) | 1× RTX PRO 6000 BW | CUDA/bf16/AMP/checkpoint on **6.2M only** |
| 6.8B train (November+) | **2× H200 141 GB** or **1× B300 ~288 GB** | AdamW bf16 ≈ 16–20 B/param → **~110–136 GB** weights+opt+grad before activations |
| 6.8B infer / Yunmu dry-run | 1× 6000 BW 96 GB or 1× H100 80 GB | weights bf16 ≈ 13.6 GB + world/activation headroom |
| Do not | 1× H100 80 GB train | will OOM on optimizer |

**6.8B train layout (target):**

```text
FSDP2 (or ZeRO-3) across 2× H200
  shard params + grads + Adam
  bf16 compute, fp32 master
  activation checkpointing on each CognitiveBlock
  cognition_budget 4 kept (do not raise to hide memory)
```

World slots stay **512**. Increasing slots to “use 96 GB” blows `N²` attention
before it helps.

Warmcore may later host the same topology for joint finetune. NULLXES keeps
architecture identity in the `*.mina` manifest.

---

## 2. Memory budget (one 6.8B replica, order of magnitude)

```text
params bf16                         ~13.6 GB
AdamW (fp32 m,v + fp32 master)    ~80–95 GB   (sharded under FSDP)
grads bf16                          ~13.6 GB   (sharded)
activations, budget=4, N=512, d=4096
  + checkpointing                   ~10–25 GB / GPU   (measure, do not guess)
world + memory buffers              << 1 GB
reserve                             15%
```

Single-GPU train is **B300-class** or 2×H200. Log `allocated` / `reserved` into
the run report the same way as `docs/GPU_BRINGUP_6000BW.md`.

---

## 3. Optimizer

```text
AdamW
lr:           1e-4  (muP-style: do not copy 6.2M lr blindly)
weight_decay: 0.01  (on DWC/NPF; 0 on NPF source table if it overfits IDs)
grad_clip:    1.0
precision:    bf16  (AMP); fp32 reductions
scheduler:    warmup 2k steps → cosine
```

Shard optimizer state. Checkpoint **resume** is mandatory: `*.mina` must reload
system + optimizer + `RuntimeState` cursor + dataset epoch index. A 6.8B run
that cannot resume is not a 6.8B run.

---

## 4. Dataset format

Not text. Not tokens. Native episode JSON from SyntheticWorld Dataset v1
(`docs/DATASET_V1.md`), streamed:

```text
episode
  observations
  world_states / teacher visibility
  actions
  future_branches
  events          (occlusion ≠ out_of_range; disappearance is real)
  outcomes / corrections
```

**Streaming:** do not preload millions of episodes. Shard by
`(split, scenario, seed, episode_index)`. Replay identity: same tuple → same
canonical JSON.

Curriculum is **experience density**, not file count:

```text
walk / coast / stop
fall / recover
grasp / drop
collision / near-miss
sensor delay / missing / noise
mass / inertia change (sim)
body variant (workspace, 170–180 cm embodiment fields)
environment variant (floor, clutter, lighting later — not vision foundation)
```

Warmcore joint path: they finetune on NULLXES episode packs + their humanoid
logs mapped into `Observation` / `ActionIntent`. They do not replace NPF/DWC
with an LLM.

---

## 5. Curriculum stages (toward 6.8B, not a second architecture)

| Stage | Data | Gate |
|---|---|---|
| A | 6.2M instrument, SyntheticWorld v1 | GPU bring-up report |
| 0 | 6.8B, 16–64 deterministic episodes | overfit: finite grads, replay, WAIT≠MOVE_TO |
| 1 | observation → belief | existence + xy NLL, persistence |
| 2 | S_t → S_{t+1} | temporal + future ADE/FDE |
| 3 | delayed / occluded / gone_forever | memory_effect_delta ≠ 0 |
| 4 | noise, conflict, OOD combos | calibration, not collapse |
| 5 | multi-future + strategy | counterfactual separation |
| 6 | hard constraints adversarial | kernel cannot be bought by value |
| 7 | closed-loop sim | ActionIntent changes next observation |
| 8 | humanoid **sim** adapter | 170–180 cm SelfModel; still no raw PWM |
| 9 | Yunmu review package | docs + checkpoint + limits |

Loss decrease is logged. It is not a gate.

---

## 6. Validation gates (must beat 6.2M instrument, not just “bigger”)

```text
Belief      correct state ↑     belief_revision_accuracy, existence
Memory      with > without      memory_effect_delta
Action      WAIT future ≠ MOVE  action_influence, counterfactual_separation
OOD         unseen combo        composition / ood splits
Uncertainty calibrated          uncertainty_calibration_error
Constraints hard                constraint_violation_count
```

If 6.8B is flatter than 6.2M on these, **stop scaling** and inspect data, not
depth.

---

## 7. Yunmu integration package

Pilot first. Full 6.8B walk on a 170–180 cm body is not the first drop.

**Now (review):**

```text
models/MINA-6.8B/architecture.yaml
docs/ARCHITECTURE.md
docs/GATE_04_IDENTITY.md      SelfModel, Authority
docs/CONSTRAINTS.md           hard > policy
docs/GATE_09_RUNTIME.md       cycle(), RuntimeState
docs/MINA_6_8B_TRAINING.md    this file
Python Observation / ActionIntent
fail-closed: MANUAL / SAFE_HOLD / ADVISORY
```

**Later (joint solution, if Warmcore finetunes):**

```text
*.mina  6.8B  + optimizer-free infer
embodiment adapter: Yunmu sensors → Observation
Yunmu controller: ActionIntent → their locomotion/manipulation stack
NULLXES constraints remain hard
```

MINA never emits motor PWM. Humanoid height 170–180 cm is embodiment metadata
and workspace limits inside SelfModel, not a new network.

---

## 8. What today on RunPod is

```text
1× RTX PRO 6000 BW
gpu_train_v01  6.2M
prove: CUDA, bf16, AMP, dataloader, .mina save/load, metrics
fill docs/GPU_BRINGUP_6000BW.md
STOP
```

Do not construct `MinakanushiSystem` from `minakanushi_6_8b.yaml` on that pod.

---

Maga: this specification is the 6.8B training contract until revised.
