# MINAKANUSHI 6.8B PRE-TRAIN GATE

**Status:** executable. Architecture frozen at `7aba976`.
**Profile:** `minakanushi_6_8b` / `models/MINA-6.8B`
**Contract:** `docs/MINA_6_8B_TRAINING.md`

This is not a month of training. This is not Yunmu. This is not a new module.

Goal: first contractual 6.8B run without architectural surprises.

```text
7aba976
  ↓
Freeze MINA foundation
  ↓
Prepare 6.8B training stack
  ↓
Episode curriculum generation
  ↓
H200 / B300 sanity pretrain
  ↓
MINA-6.8B checkpoint
  ↓
only then Yunmu humanoid
```

## Frozen cognitive scheme

Scale this MINA. Do not search for another architecture.

```text
Identity ✅
Authority ✅
Belief ✅
Memory ✅
Focus ✅
Active World Model ✅
Runtime cycle ✅
Revision learning ✅
6.8B target profile
```

Frozen dimensions (do not edit):

```text
latent_dim: 4096
state_dim: 4096
memory_dim: 4096
world_slots: 512
memory_slots: 1024
core_depth: 32
cognition_budget: 4
```

Forbidden until Maga revises this gate:

```text
48 layers
more slots
new learned module
λ retune as a substitute for data
construct 6.8B on CPU
construct 6.8B on RTX PRO 6000
train 6.8B on 1× H100 80GB
FP16
token datasets
Yunmu / Gate 10
```

## Stack that must exist before H200

```text
FSDP2 / ZeRO-3          minakanushi/training/parallel.py
bf16 compute            fp32 reductions
activation checkpoint   CognitiveBlock only
sharded *.mina          resume + validation restore
episode curriculum      not tokens
sanity_pretrain.py      refuses 6.8B construct on the wrong machine
```

Sanity pretrain looks at:

```text
loss is finite
gradients finite
checkpoint save / resume
belief_revision_accuracy
memory_effect_delta
future branch quality
uncertainty calibration
causal consistency
```

`loss ↓` is logged. It is not the gate.

## Dataset

Episodes, not tokens:

```text
observation_t → belief_t → action_t → world transition
  → observation_t+1 → correction → lesson
```

Phases: physics, agency, causality. Embodiment metadata is passport
fields (170–180 cm workspace). Not a humanoid network. Not PWM.

## Hardware

Blackwell 6000 BW is closed for this gate (instrument work is done).

Train 6.8B on **2× H200** or **1× B300**. Infer later may use 6000 BW.

## Operator

```text
python scripts/sanity_pretrain.py
python scripts/generate_6_8b_curriculum.py --root dataset/mina_6_8b --n 2
```

Do not `python scripts/train.py --config configs/training/mina_6_8b_sanity.yaml`
on a laptop or the 6000 BW pod. That YAML is the H200/B300 contract.
