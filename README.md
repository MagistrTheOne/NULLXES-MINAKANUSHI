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
| research_v01 | `configs/architecture/research_v01.yaml` | 2048 | 512 | 24 | canonical research target |
| gpu_train_v01 | `configs/architecture/gpu_train_v01.yaml` | 256 | 64 | 6 | GPU training |
| cpu_dev | `configs/architecture/cpu_dev.yaml` | 64 | 16 | 2 | tests / CPU loop |

All dimensions are configuration. Model code must not assume 2048.

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
- `docs/CONSTRAINTS.md`
- `docs/GATE_03_PRE_WORLD_MODEL.md` — approved lock: belief target, SelfModel
  as passport not network, authority gates action not cognition
- `docs/GATE_04_IDENTITY.md` — SelfModel, Authority, Persona (not a prompt)
- `docs/GATE_05_BELIEF.md` — Belief Engine (mean + std + existence)
- `docs/GATE_06_EXPERIENCE.md` — memory as experience (not RAG)
- `docs/GATE_07_FOCUS.md` — Focus Engine / internal attention (not desire)
- `docs/GATE_08_WORLD_MODEL.md` — Belief(t)+Action → Belief(t+1)
- `docs/GATE_08_5_DATASET.md` — Dataset Reality Check (not Gate 09)
- `docs/DATASET_V1.md` — SyntheticWorld dataset contract
- `docs/RUNPOD.md` — GPU pipeline order (not started)
- `docs/PRETRAINING_GATE_01.md`
