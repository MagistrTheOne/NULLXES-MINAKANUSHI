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
  - not-a-llm
  - researched
base_model: []
---

# MINAKANUSHI-6.8B

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

| File | What |
|---|---|
| [`minakanushi_stage0_step64.mina`](minakanushi_stage0_step64.mina) | Status Core v0.1 · 1× H200 · seed 11 · git `d70bfc0` · 64 steps |
| [`minakanushi_stage0_step128.mina`](minakanushi_stage0_step128.mina) | Status Core v0.2 · 1× B300 · resume IdentityBound · JSON curriculum · steps **65–128** · git `ede6bda` |

Format: native `nullxes-minakanushi` zip (`ZIP_STORED`). Same architecture. v0.2 is a continuation (optimizer + identity), not a clone.

Safetensors shards are a **later** Hugging Face mirror (Hub parameter badge). They are not the runtime. Canonical load stays `load_mina`. Do not convert step64. See `docs/HF_SAFETENSORS_MIRROR.md`.

Also at root: `metrics_v02.jsonl`, `train_v02.log`.

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

- [ ] Acceptance Gate v0.2 — `scripts/gate_v02_acceptance.py` (`cpu_dev` first)
- [ ] Yunmu review — ActionIntent in, their controller out; not before the gate
- [ ] HF safetensors mirror — after Acceptance Gate; `.mina` stays canonical; see `docs/HF_SAFETENSORS_MIRROR.md`
- [ ] Long 6.8B — same freeze, same native JSON
- [ ] Gate 9+ perception — pixels → MinaUnit
- [ ] MINA V2 MM — organs → MinaUnit (not VLA/Cosmos); not train until Yunmu/Gate 9+

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
