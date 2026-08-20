# NULLXES MINAKANUSHI

Adaptive situational intelligence for autonomous physical systems.

This repository is the native MINAKANUSHI runtime, not a wrapper around another
foundation model. Identity lives in architecture config, checkpoint manifests,
module names, and the `*.mina` format.

```text
architecture: MINAKANUSHI
organization: NULLXES
system_class: adaptive_situational_intelligence
native_runtime: nullxes
architecture_version: 0.1
```

## Clean install

```text
python -m pip install -e ".[test]"
python -m pytest tests
python scripts/generate_dataset.py --root dataset --n 4 --length 8
```

CPU only. Do not instantiate `minakanushi_6_8b` or `research_v01` in tests.
Checkpoints are `*.mina` and are not committed.

## What exists now (Milestone 1 foundation)

Implemented and wired into one closed loop:

```text
SyntheticWorld → Observation → MinaUnit → NPF → StateConstructor
 → DynamicWorldCore → Memory + Uncertainty → Situation
 → FutureEngine → StrategyEngine → ConstraintKernel
 → ActionPolicy → ActionIntent → SyntheticWorld
```

Learned substrate: NullxesPositionField, perception encoders, DynamicWorldCore
slot updater, uncertainty head, future residual.

Non-learned authority: MinakanushiConstraintKernel. Hard constraints are
evaluated before policy selection and cannot be overridden by value.

## Config profiles

| Profile | Path | latent | slots | depth | Role |
|---|---|---:|---:|---:|---|
| **minakanushi_6_8b** | `configs/architecture/minakanushi_6_8b.yaml` / `models/MINA-6.8B/` | 4096 | 512 | 32 | **Yunmu contract · 6.8B · [HF Status Core](https://huggingface.co/MagistrTheOne/MINAKANUSHI-6.8B)** |
| research_v01 | `configs/architecture/research_v01.yaml` | 2048 | 512 | 24 | 1.3B rung, not the product target |
| gpu_train_v01 | `configs/architecture/gpu_train_v01.yaml` | 256 | 64 | 6 | GPU bring-up instrument (6.2M) |
| cpu_dev | `configs/architecture/cpu_dev.yaml` | 64 | 16 | 2 | tests / CPU loop |

All dimensions are configuration. Model code must not assume 2048 or 4096.

## Training

Composite objective, not next-token:

```text
L = λs L_state + λt L_temporal + λf L_future + λu L_uncertainty
  + λc L_causal + λm L_memory + λa L_action + λr L_representation
```

Curriculum:

- Stage 0 architecture validation — `configs/training/stage0_validation.yaml`
- Stage 0 overfit wiring proof — `configs/training/stage0_overfit.yaml` (16 deterministic episodes)
- Stage 0 generalization eval (no training) — `configs/training/stage0_generalization.yaml`
- Stage 1 world representation — `configs/training/stage1_world.yaml`
- Stage 2 temporal dynamics — `configs/training/stage2_temporal.yaml`
- 6.8B sanity contract — `configs/training/mina_6_8b_sanity.yaml` (H200/B300 only)

CPU stack check (does **not** construct 6.8B):

```text
python scripts/sanity_pretrain.py
python scripts/generate_6_8b_curriculum.py --root dataset/mina_6_8b --n 2
```

Commands (operator-authorized machines only):

```text
python scripts/train.py --config configs/training/stage0_overfit.yaml
python scripts/evaluate.py
python scripts/simulate.py --steps 40
python -m pytest tests
```

Checkpoints are native `*.mina` zip archives (YAML manifest + tensors).

## Applications (intended)

- mobile inspection / patrol robots
- warehouse AMRs under human no-go zones
- simulation-first world-state research
- later: controlled physical platforms after Stage 9

Not in scope: chat, coding assistants, RAG agents, wrapping Qwen/Llama/GPT.

## Docs

- `docs/ARCHITECTURE.md`
- `docs/POSITION_FIELD.md`
- `docs/WORLD_STATE.md`
- `docs/MEMORY.md`
- `docs/TRAINING.md`
- `docs/TRAINING_PLAN.md` — 6.8B pre-train order after 7aba976 freeze
- `docs/GPU_BRINGUP_6000BW.md` — Stage A report
- `docs/CONSTRAINTS.md`
- `docs/GATE_6_8B_PRETRAIN.md` — current lock: FSDP2, bf16, episode curriculum
- `docs/GATE_03_PRE_WORLD_MODEL.md` — closed: belief target, SelfModel
  as passport not network, authority gates action not cognition
- `docs/GATE_03A_BELIEF_REVISION.md` — constructor revision primitives
- `docs/GATE_03_REVISION_VALIDATION.md` — training-loop exam (DWC vs evidence)
- `docs/GATE_03B_HIDDEN_DIRECTION.md` — hidden 0.5: prior vs evidence diagnostic
- `docs/GATE_03B_HIDDEN_DIRECTION_REPORT.md` — closed, Variant 1, n=1000
- `docs/GATE_04_IDENTITY.md` — SelfModel, Authority, Persona (not a prompt)
- `docs/GATE_05_BELIEF.md` — Belief Engine (mean + std + existence)
- `docs/GATE_06_EXPERIENCE.md` — memory as experience (not RAG)
- `docs/GATE_07_FOCUS.md` — Focus Engine / internal attention (not desire)
- `docs/GATE_08_WORLD_MODEL.md` — Belief(t)+Action → Belief(t+1)
- `docs/GATE_08_5_DATASET.md` — Dataset Reality Check (not Gate 09)
- `docs/GATE_09_RUNTIME.md` — Autonomous Runtime (`cycle()`, RuntimeState)
- `docs/DATASET_V1.md` — SyntheticWorld dataset contract
- `docs/MINA_6_8B_TRAINING.md` — 6.8B contract (Yunmu / Warmcore)
- `docs/RUNPOD.md` — RTX 6000 BW Stage A (`gpu_train_v01` instrument)
- `docs/PRETRAINING_GATE_01.md`
