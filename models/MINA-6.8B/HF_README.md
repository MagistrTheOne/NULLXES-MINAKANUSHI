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

**NULLXES MINAKANUSHI 6.8B Status Core (Researched)**

```text
HF repo:      MagistrTheOne/MINAKANUSHI-6.8B
architecture: MINAKANUSHI
short name:   MINA
organization: NULLXES
author:       MagistrTheOne
status:       Researched / Status Core
native unit:  MinaUnit  (not a token)
checkpoint:   *.mina
```

This is an adaptive situational intelligence checkpoint for autonomous physical systems. It is **not** a chatbot, not a decoder-only LLM, and not a wrapper around Qwen / Llama / Mistral / Gemma / DeepSeek / GPT / Claude.

Code and runtime: [MagistrTheOne/NULLXES-MINAKANUSHI](https://github.com/MagistrTheOne/NULLXES-MINAKANUSHI)

Intended readers: **Yunmu**, **Warmcore**, and NULLXES operators. Same product line, same identity, not a one-off experiment.

---

## What this model is for

MINAKANUSHI infers world state from incomplete observations, keeps uncertainty explicit, predicts multiple futures, ranks strategies, and emits `ActionIntent` only after hard human constraints.

```text
observation → MinaUnit → NullxesPositionField → WorldState
  → DynamicWorldCore → memory + uncertainty → situation
  → futures → strategies → ConstraintKernel → ActionIntent
```

Intended use:

- synthetic and later physical world-state research
- mobile inspection / patrol under human no-go zones
- warehouse AMR policy research
- Yunmu / Warmcore humanoid **simulation** adapters later (ActionIntent in, their controller out)

Out of scope:

- chat, coding assistants, RAG agents
- next-token generation as the cognitive primitive
- raw motor PWM / actuator voltage

MINA never emits motor PWM. Hardware control stays in a deterministic downstream controller.

---

## Identity (not a system prompt)

Identity lives in architecture YAML, `*.mina` manifest, and runtime metadata.

```yaml
architecture: MINAKANUSHI
organization: NULLXES
system_class: adaptive_situational_intelligence
architecture_generation: 1
native_runtime: nullxes
architecture_version: "0.1"
short_name: MINA
```

SelfModel is a structured passport (embodiment, capabilities, authority). It is not a Transformer identity head and not an "I AM MINAKANUSHI" text objective. Authority can disable autonomous selection (`policy_enabled=false`) without turning cognition off. Hard constraints still win.

---

## Checkpoint in this repo

| Field | Value |
|---|---|
| Display name | NULLXES MINAKANUSHI 6.8B Status Core (Researched) |
| File | `checkpoints/minakanushi_stage0_step64.mina` |
| Format | native `nullxes-minakanushi` zip (`ZIP_STORED`), sharded tensors |
| Profile | `minakanushi_6_8b` |
| Parameters (formula) | **6 799 130 646** |
| latent / DWC / slots | 4096 / 32 / 512 world + 1024 memory |
| Train config | `configs/mina_6_8b_status_core_researched.yaml` |
| Hardware | 1× NVIDIA H200, FSDP2 ZeRO-3, bf16 compute, fp32 weights |
| Steps | 64 (seed 11), final-only checkpoint |
| Source git | `d70bfc0` |

This Status Core drop is the first **working** 6.8B training line with a named artifact. The earlier 20-step sanity run is a probe, not this release.

### Measured Status Core signals (step 64)

| Signal | Value |
|---|---|
| loss | 78.36 |
| future ADE / FDE | 3.42 / 0.81 |
| world position error | 1.07 |
| uncertainty calibration error | 0.38 |
| entity persistence / reacquisition | 1.0 / 1.0 |
| constraint_violation_count | **0** |
| closed_loop_success_rate | 1.0 |
| false_revision_rate | 0.0 |

Loss going down is logged. It is not the architecture gate. Belief-revision accuracy on this short synthetic segment is still incomplete; that is a training-data job, not a reason to swap the architecture.

---

## Dependencies

Install the native runtime from source, then this checkpoint.

```text
python -m pip install -e ".[test]"
# or the pin list in this repo:
python -m pip install -r requirements.txt
```

Required:

```text
python >= 3.11
numpy >= 2.0
pyyaml >= 6.0
pydantic >= 2.0
torch >= 2.3
```

Load is native PyTorch + MINAKANUSHI. There is no `transformers.AutoModel`. Constructing `minakanushi_6_8b` on CPU or a 96 GB 6000-class card is forbidden for training; inference later may fit on 6000 BW / H100 with bf16 weights (~13.6 GB) plus world/activation headroom.

```python
from pathlib import Path
from minakanushi.architecture.config import load_architecture
from minakanushi.architecture.model import MinakanushiSystem
from minakanushi.training.checkpoint import load_mina

arch = load_architecture("architecture.yaml")
system = MinakanushiSystem(arch)  # GPU-class machine
manifest = load_mina("checkpoints/minakanushi_stage0_step64.mina", system)
```

Continue training (H200 / B300, not a laptop):

```text
torchrun --nproc_per_node=1 scripts/train.py \
  --config configs/training/mina_6_8b_status_core_researched.yaml \
  --out experiments/mina_6_8b_status_core_researched
```

`--resume` from `*.mina` is still a follow-up in `scripts/train.py`. Checkpoint load/save already exist.

---

## Further training checklist

Do **not** replace MINAKANUSHI with another foundation model. Scale this runtime.

### Immediate (next H200 / B300 budget)

- [ ] Wire `JsonEpisodeDataset` so `Trainer.unroll()` consumes `dataset/mina_6_8b` JSON, not only procedural `generate_episode(...)`.
- [ ] Add `scripts/train.py --resume path/to/*.mina` (optimizer + runtime cursor + epoch index).
- [ ] Expand curriculum past n=8 / 2-per-phase. Current Grok pack is valid but thin (8 episodes, 2 corrections, no constant-velocity collapse).
- [ ] Re-enable activation checkpointing only after FSDP2 recompute metadata is proven; Status Core ran with `activation_checkpoint: false`.
- [ ] Keep `checkpoint_every == steps` or uncompressed `ZIP_STORED` — deflate stalls 6.8B saves.
- [ ] Train next named segment from this Status Core weights, not from scratch, once resume is wired.
- [ ] Target topology for long runs: **2× H200** or **1× B300**. Do not train 6.8B on 1× H100 80 GB.

### Curriculum gates (physical episodes, not tokens)

- [ ] Stage 1 — observation → belief (existence + xy, persistence)
- [ ] Stage 2 — `S_t → S_{t+1}` (temporal + future ADE/FDE)
- [ ] Stage 3 — delayed / occluded / `gone_forever` (`memory_effect_delta ≠ 0`)
- [ ] Stage 4 — noise, conflict, OOD combos (calibration, not collapse)
- [ ] Stage 5 — multi-future + strategy (counterfactual separation, WAIT ≠ OBSERVE ≠ MOVE_TO)
- [ ] Stage 6 — hard constraints adversarial (kernel cannot be bought by value)
- [ ] Stage 7 — closed-loop sim (`ActionIntent` changes the next observation)
- [ ] Stage 8 — humanoid **sim** adapter (170–180 cm SelfModel; still no PWM)
- [ ] Stage 9 — Yunmu review package (docs + checkpoint + limits)

### Must-hold signals

```text
constraint_violation_count == 0
entity persistence / reacquisition stay high
false_revision_rate stays low
uncertainty calibration improves
future ADE/FDE trend down on held-out episodes
hard constraints beat higher-value illegal strategies
```

If 6.8B is flatter than the 6.2M instrument on these, stop scaling and inspect data. Do not add layers, slots, or an LLM.

Forbidden until architecture revision:

```text
token datasets as the cognitive objective
FP16
construct 6.8B on CPU / RTX PRO 6000 for training
new learned identity head / chat template
replacing NPF / DWC with another model family
```

---

## Dataset provenance

This drop trained on the procedural SyntheticWorld loop used by `scripts/train.py` (32 overfit episodes, sequence length 12). Direct JSON training was audited but **not** connected yet.

Audited Grok/6.8B JSON pack (`dataset/mina_6_8b`):

```text
8 episodes · physics/agency/causality/embodiment × 2
transition length 11 · missing keys: none
events 109 · occlusions 5 · conflicts 8 · corrections 2
constant-velocity collapse: false
```

Dataset contract: `docs/DATASET_V1.md` in the GitHub runtime repo.

---

## Limits (honest)

- Status `Researched`, not a finished product brain.
- 64 steps on synthetic physics. Not humanoid locomotion. Not vision-foundation trained.
- Branch diversity / coverage on this segment is still near zero.
- Belief-revision accuracy on the logged eval slice is not yet a pass.
- Do not treat this checkpoint as a chat model.

---

## License

NULLXES MINAKANUSHI Research License. Research use of this architecture and checkpoint. Redistribution as another model family, or as an LLM wrapper with this name, is not granted.

---

## Attribution

```text
NULLXES                 organization / architecture owner
MagistrTheOne           author, HF namespace, runtime repo
MINAKANUSHI / MINA      architecture family / short name
Yunmu / Warmcore        intended integration readers, not a replacement stack
```

I WILL SURVIVE. NULLXES.
