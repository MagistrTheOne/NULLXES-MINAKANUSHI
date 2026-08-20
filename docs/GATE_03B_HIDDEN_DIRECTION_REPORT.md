# Gate 03B — Hidden direction report

**Status:** empty until Blackwell `n=1000`. Do not fill from one CPU seed.
**Protocol:** `docs/GATE_03B_HIDDEN_DIRECTION.md`
**Profile:** `gpu_train_v01`. Not 6.8B.

Copy numbers from `experiments/gate03b/gate03b_hidden_direction.json`.
Loss is not the verdict.

```text
Constructor:  evidence > old belief, CorrectionEvent exists
DWC:          prior dynamics strong, residual cautious
Result:       revision detected = yes; direction = the question
```

## Run

```text
date:
pod:
git: main
n_per_class: 1000
training: configs/training/stage_a_gpu_train_v01.yaml
checkpoint:
lambda_revision: 1.0
```

## hidden_correction

```text
scenario: hidden_correction
n:
mean:
median:
std:
min:
max:
false_revision:
revision_latency:
detected:
magnitude_error:
state_over_revision:
future_over_revision:
```

## conflict

```text
scenario: conflict
n:
mean:
median:
std:
false_revision:
```

## reacquisition

```text
scenario: reacquisition
n:
mean:
median:
std:
false_revision:
identity_same_hypothesis:
```

## Decision (pick one)

```text
[ ] Variant 1 — hidden mean ≥ 0.7, false_revision ≈ 0
    signal reached, capacity enough. Close Gate 03.

[ ] Variant 2 — hidden mean stuck ~0.5, false_revision ≈ 0
    physics prior vs evidence. Do not bump λ.
    Next: term ratios, then causal correction objective
    ("why the forecast was wrong"), not a bigger model.

[ ] Variant 3 — direction ↑ and false_revision ↑
    STOP. That is amnesia ("new always beats old"), not intelligence.
    Need: evidence > hypothesis only when evidence is reliable.
```

```text
verdict:
gate03_closed:
next:
operator:
```
