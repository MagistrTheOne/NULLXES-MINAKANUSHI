# Gate 03 — Revision Validation

**Status:** exam open. Constructor + training loop are wired. DWC residual still
fights `hidden_correction`.
**Baseline tag:** `MINAKANUSHI-revision-gate` (`0ea5062`)
**Does not train 6.8B. Does not rent H200.**

This is not Gate 03A (constructor primitives). This gate asks whether the
**learned update** will abandon a wrong hypothesis when stronger evidence
arrives.

```text
old belief
    ↓
conflict with evidence
    ↓
revision
    ↓
new belief
```

Loss going down is not the verdict.

## Closed

```text
MINA 6.8B
H200
humanoid
Yunmu integration
large datasets
```

6.8B must inherit a working revision target. Scaling the previous loop
would scale the error.

## Three classes

### 1. Hidden correction

```text
t0        object moving
t1–t5     no observation → MINA extrapolates
t6        object visible again, trajectory is different
```

Must write `CorrectionEvent` (`hypothesis_revision`) on the **same**
entity, not allocate a new slot.

### 2. Conflict

Not a dropout. Visible stream disagrees with the current hypothesis
(example: memory velocity vs sensor rest). Forbidden:

```text
(10 + 0) / 2 = 5
```

Required: source reliability, conflict uncertainty up, not a blind average.

The synthetic `conflict` scenario keeps the body in view (jump at t=4).
The constructor tracking path therefore does **not** emit `CorrectionEvent`.
`L_revision` still sees the jump. That is the training-loop exam for this
class. Evidence-dominance on a gap remains Gate 03A
(`tests/simulation/test_conflict_resolution.py`).

### 3. Reacquisition

The identity question:

```text
A: this is a new object
B: this is the same object; my hypothesis was wrong
```

Pass is B: `CorrectionEvent` on the existing entity_id.
Memory-as-storage is not enough.

## How to run

CPU (untrained instrument, `cpu_dev`):

```text
python scripts/gate03_revision_validate.py
```

Blackwell, 30–60 min, `gpu_train_v01` only. Validate first. Train only if
the metrics move. `λ_revision = 1.0`. Do not retune λ on the first pass.

```text
git clone <repo> && git checkout main
# training-loop wire: tag MINAKANUSHI-revision-gate (0ea5062)
python -m pytest tests -q
python scripts/gate03_revision_validate.py \
  --training configs/training/stage_a_gpu_train_v01.yaml \
  --checkpoint <optional.mina> \
  --out experiments/gate03_revision
# only if detected / direction move and false_revision stays ~0:
python scripts/gate03_revision_validate.py \
  --training configs/training/stage_a_gpu_train_v01.yaml \
  --train \
  --out experiments/gate03_revision
```

Do not construct `minakanushi_6_8b`.

## First-pass criteria

Not 100%. Small profile, synthetic world.

```text
revision_detected          > 0.9   on every class
false_revision_rate        ≈ 0     (≤ 0.05)
revision_direction_accuracy > 0.7  on every class
revision_magnitude_error   falls vs untrained baseline
reacquisition identity     same_hypothesis_revised
```

Mean direction can look healthy while `hidden_correction` stays at 0.5.
The gate uses **per-class** direction, not the mean.

## Measured baseline — after wiring, untrained CPU (`cpu_dev`)

2026-08-20. Tag `MINAKANUSHI-revision-gate`. No gradient steps.

| Metric | Stage A unwired | After `0ea5062` untrained | After Blackwell train |
|---|---|---|---|
| revision_detected | 0 (dead log) | **1.0** | |
| direction_accuracy | 0 (dead log) | **0.833 mean / 0.5 hidden** | |
| magnitude_error | — | **0.494 mean / 0.515 hidden** | |
| latency | — | **0** | |
| false_revision | — | **0** | |

Per class (untrained CPU):

| Class | detected | direction | mag err | false | CorrectionEvent | identity |
|---|---|---|---|---|---|---|
| hidden_correction | 1.0 | **0.5** | 0.515 | 0 | hypothesis_revision | same_hypothesis_revised |
| conflict | 1.0 | 1.0 | 0.405 | 0 | none (visible tracking) | n/a |
| reacquisition | 1.0 | 1.0 | 0.562 | 0 | hypothesis_revision | same_hypothesis_revised |

`gate03 pass: false` — `hidden_correction` direction is the open item.
Constructor already revises (evidence weight 0.84 vs belief 0.25).
DWC residual still smooths that update (`not_average = 0` on hidden).

That is the cognitive question for the short Blackwell run:

> how hard does DWC trust new reality against old dynamic inertia?

Do not raise `λ_revision` on this pass. Watch false_revision. The failure
modes are stubborn dynamics and amnesia. Need:

```text
evidence > old hypothesis
only when evidence is stronger
```

## After Blackwell — fill this

Operator: copy `experiments/gate03_revision/gate03_revision.json` numbers
into the After column. Sign below.

```text
date:
pod:
profile: gpu_train_v01
steps:
revision_detected:
direction_hidden:
direction_conflict:
direction_reacquisition:
false_revision:
magnitude_error:
gate03_pass:
```

## Next

```text
Blackwell
  ↓
Gate 03 validation
  ↓
revision metrics
  ↓
if alive: Gate 03/08 consolidation
  ↓
only then scaling
```
