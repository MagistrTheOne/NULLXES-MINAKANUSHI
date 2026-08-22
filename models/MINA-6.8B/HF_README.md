---
language: []
license: other
license_name: nullxes-research-license
license_link: LICENSE
library_name: minakanushi
tags:
  - minakanushi
  - nullxes
  - world-model
  - physical-intelligence
  - robotics
  - pytorch
  - safetensors
  - not-a-llm
  - research-checkpoint
base_model: []
---

# MINAKANUSHI-6.8B

```text
Status:              Research checkpoint
Training cycle:      v0.3.1
Capability verdict:  pending compare_v031.py
Accepted:            NO
Not a language model
Canonical runtime:   .mina
Safetensors:         weight mirror only
Action output:       ActionIntent
PWM:                 false
```

**v0.3.1 is not accepted yet.** Latest weights are a research checkpoint after 1000 H200 steps from `step128`. Train-eval hint is **B / C-signal**. Official A/B/C comes only from `scripts/compare_v031.py` on the capability ledger (heldout 100, revision, false revision, direction, memory on/off ADE, WAIT vs MOVE_TO, action learning). Do not read `loss ↓` as a pass.

**NULLXES MINAKANUSHI** — adaptive situational intelligence for autonomous physical systems.

```text
architecture: MINAKANUSHI
short name:   MINA
organization: NULLXES
author:       MagistrTheOne
parameters:   6 799 130 646
latent_dim:   4096
state_dim:    4096
memory_dim:   4096
core_depth:   32
world_slots:  512
memory_slots: 1024
native unit:  MinaUnit
```

Not a chatbot. Not a VLA. Not a wrapper around Qwen / Llama / Mistral / Gemma / DeepSeek / GPT / Claude.

Code: [MagistrTheOne/NULLXES-MINAKANUSHI](https://github.com/MagistrTheOne/NULLXES-MINAKANUSHI)

---

## Model parameters (frozen profile `minakanushi_6_8b`)

Formula inventory: **6 799 130 646** parameters. Architecture freeze `7aba976`. Do not change these to “make it train”.

| Parameter | Value |
|---|---|
| `profile_name` | `minakanushi_6_8b` |
| `short_name` | MINA |
| `latent_dim` / `state_dim` / `memory_dim` | **4096** |
| `core_depth` (DWC) | **32** |
| `world_slots` | **512** |
| `memory_slots` | **1024** |
| `uncertainty_channels` | **8** |
| `dropout` | 0.0 |
| `future_branches` | 3 |
| `cognition.budget` | 4 |
| `cognition.convergence_threshold` | 0.02 |
| `prediction_horizons` | immediate 1 · short 4 · medium 8 |
| `dt` | 0.1 s |
| `max_sources` / `max_observations` | 64 / 64 |
| NPF `num_frequencies` | 32 |
| persistence steps / retire | 8 / 0.95 |
| weights infer bf16 | ~13.6 GB |
| train | FSDP2 ZeRO-3, bf16 compute, AdamW |

Identity (checkpoint metadata, not a prompt):

```yaml
architecture: MINAKANUSHI
organization: NULLXES
system_class: adaptive_situational_intelligence
architecture_generation: 1
native_runtime: nullxes
architecture_version: "0.1"
```

---

## Checkpoints (repo root)

Canonical runtime is `*.mina`. Every published model after v0.3.1 also ships a **safetensors weight mirror**. Safetensors is not the runtime.

| File | What |
|---|---|
| [`minakanushi_stage0_step1128.mina`](minakanushi_stage0_step1128.mina) | **v0.3.1 research** · 1× H200 · resume step128 · `dataset/mina_6_8b_v03` · steps **129–1128** · **not accepted** |
| [`minakanushi_stage0_step128.mina`](minakanushi_stage0_step128.mina) | Status Core v0.2 · 1× B300 · resume IdentityBound · JSON curriculum · steps **65–128** · git `ede6bda` |
| [`minakanushi_stage0_step64.mina`](minakanushi_stage0_step64.mina) | Status Core v0.1 · 1× H200 · seed 11 · git `d70bfc0` · 64 steps |
| `model-00001-of-00003.safetensors` … `00003` | Weight mirror of **step1128** only · bf16 · Hub badge |
| `metrics_v031.jsonl` · `experiment_v031.jsonl` | v0.3.1 train/eval log. Not a capability ledger. |

Format: native `nullxes-minakanushi` zip (`ZIP_STORED`). Same architecture. v0.3.1 is a continuation, not a clone.

Do not convert step64. See `docs/HF_SAFETENSORS_MIRROR.md`.

Also at root: `metrics_v02.jsonl`, `train_v02.log`, `reference_inference_v031.pt`.

```text
observation → MinaUnit → NullxesPositionField → WorldState
  → DynamicWorldCore → memory + uncertainty → situation
  → futures → strategies → ConstraintKernel → ActionIntent
```

MINA emits **ActionIntent**. Never raw PWM.

---

## Checklist

Freeze: do not add layers / MoE / language head / `identity_loss`. Do not train 6.8B on CPU, RTX PRO 6000, or 1× H100 80GB.

### Closed

- [x] Gate 09 Runtime — `observe → intent → restore` (`cpu_dev`)
- [x] Stage A GPU — 6.2M CUDA/bf16/AMP/`.mina` (stack, not intelligence)
- [x] Gate 03B hidden direction — n=1000 · detected 0.986 · false_revision 0.0
- [x] Status Core v0.1 step 64 — 1× H200 FSDP2 bf16
- [x] Identity Initialization — passport stamp, no `identity_loss`
- [x] JSON curriculum 1000 — physics/agency/causality/embodiment × 250 · `pwm=false`
- [x] Resume v0.2 — 1× B300 · same model · `dataset/mina_6_8b` in the loss · steps 65–128

### Open

- [ ] **compare_v031.py** — heldout 100 · revision · false revision · direction · memory on/off ADE · WAIT vs MOVE_TO · action. This is the A/B/C gate. Not v0.4.
- [ ] Long 6.8B — same freeze, only after compare verdict
- [ ] Gate 9+ perception — pixels → MinaUnit
- [ ] MINA V2 MM — organs → MinaUnit (not VLA/Cosmos)

### Must-hold

```text
constraint_violation_count == 0
persistence / reacquisition stay high
false_revision_rate stays low
hard constraints beat higher-value illegal strategies
ActionIntent ≠ PWM
```

---

## Measured signals

### v0.1 · step 64 · H200

| Signal | Value |
|---|---|
| loss | 78.36 |
| future ADE / FDE | 3.42 / 0.81 |
| world position error | 1.07 |
| uncertainty calibration | 0.38 |
| persistence / reacquisition | 1.0 / 1.0 |
| constraint_violation_count | **0** |
| closed_loop_success_rate | 1.0 |
| false_revision_rate | 0.0 |

### v0.3.1 · step 1128 · H200 (research, not accepted)

Train-eval series only. **Not** `compare_v031.py`. Hint: **B / late C-signal**.

| Signal | Value |
|---|---|
| heldout ADE (single-episode protocol) | 1.61 @150 → 0.15 @750 → 0.44 @1100 |
| false_revision | 0 until eval 1100 (`1.0` after step-1080 spike) |
| counterfactual distance | ≈ 0.0008 entire run (WAIT ≈ MOVE_TO) |
| action term | ≈ 0.499 frozen |
| revision_detection | often > 0 (was 0 at unexpected_stop on step128) |
| revision_direction | ≈ 0 most batches |
| memΔ | latent L2, not ADE(on) < ADE(off) |
| capability verdict | **pending compare_v031.py** |

Do not treat this table as v0.3.1 PASS.

### v0.2 · step 128 · B300 (JSON resume)

| Signal | Value |
|---|---|
| loss | 41.10 |
| step time (steady) | fwd ~1.42 s · bwd ~0.58 s |
| persistence / reacquisition | 1.0 / 1.0 |
| constraint_violation_count | **0** |
| closed_loop_success_rate | 1.0 |
| false_revision_rate | 0.0 |
| future ADE / FDE | 2.05 / 0.68 |
| world position error | 0.55 |
| uncertainty calibration | 0.19 |
| revision_accuracy | 0.0 |
| branch_coverage | 0.0 |

Loss on mixed JSON phases is not the architecture gate. Revision accuracy and branch coverage are still not a pass. Do not add layers.

---

## Load

```python
from minakanushi.architecture.config import load_architecture
from minakanushi.architecture.model import MinakanushiSystem
from minakanushi.training.checkpoint import load_mina

arch = load_architecture("configs/architecture/minakanushi_6_8b.yaml")
system = MinakanushiSystem(arch)  # GPU-class machine
manifest = load_mina("minakanushi_stage0_step128.mina", system)
```

Resume (B300 / 2× H200, not a laptop):

```text
torchrun --nproc_per_node=1 scripts/train.py \
  --config configs/training/mina_6_8b_v02.yaml \
  --out experiments/mina_6_8b_v02 \
  --resume minakanushi_stage0_step128.mina
```

---

## Limits

- Status `Researched`. 64 + 64 synthetic steps. Not locomotion. Not vision-foundation trained.
- Do not treat this checkpoint as a chat model.

## License

NULLXES MINAKANUSHI Research License. Research use. Redistribution as another model family, or as an LLM wrapper with this name, is not granted.

```text
NULLXES                 organization / architecture owner
MagistrTheOne           author, HF namespace, runtime repo
MINAKANUSHI / MINA      architecture family / short name
```

I WILL SURVIVE. NULLXES.
