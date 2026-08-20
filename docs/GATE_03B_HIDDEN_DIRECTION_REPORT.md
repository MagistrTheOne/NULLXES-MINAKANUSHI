# Gate 03B — Hidden direction report

**Status:** closed. Maga accepted Blackwell `n=1000` as the Gate 03B diagnostic lock.
**Protocol:** `docs/GATE_03B_HIDDEN_DIRECTION.md`
**Profile:** `gpu_train_v01`. Not 6.8B.

Source: `/workspace/NULLXES-MINAKANUSHI/experiments/gate03b/gate03b_hidden_direction.json`
(pin SHA `7fd8ef6`; p10/p90 not in that pin). Loss is not the verdict.

```text
Constructor:  evidence > old belief, CorrectionEvent exists
DWC:          prior dynamics strong, residual cautious
Result:       revision detected = yes; direction = the question
```

## Run

```text
date: 2026-08-20 13:09 UTC
pod: gn3eqwxuht23qs  RTX PRO 6000 BW  RUNNING  ~$2.09/hr
git: 7fd8ef6 on pod; local main c152f75
n_per_class: 1000
training: configs/training/stage_a_gpu_train_v01.yaml
checkpoint: none (eval, untrained gpu_train_v01)
lambda_revision: 1.0
pid 3776: finished (GPU 0% / 0 MiB after JSON)
```

## hidden_correction

```text
scenario: hidden_correction
n: 1000
mean: 0.7605
median: 1.0
std: 0.2558
p10: not in pin 7fd8ef6
p90: not in pin 7fd8ef6
min: 0.0
max: 1.0
false_revision: 0.0
revision_latency: -0.006
detected: 0.986
magnitude_error: 1.0803
state_over_revision: 0.380
future_over_revision: 1.650
```

CPU single-seed 0.5 was a tail/seed artifact. Median 1.0 with mean 0.76:
most episodes commit; a minority stay at 0.

## conflict

```text
scenario: conflict
n: 1000
mean: 0.9185
median: 1.0
std: 0.1848
false_revision: 0.0
detected: 0.940
```

## reacquisition

```text
scenario: reacquisition
n: 1000
mean: 0.7605
median: 1.0
std: 0.2558
false_revision: 0.0
identity_same_hypothesis: 1.0
```

Not an independent class on this run. `generate_episode` maps
`reacquisition` → same physics as `hidden_correction`, and Gate 03B
reuses `episode_index=seed0+i` per class, so the worlds match hidden
byte-for-byte. Identity 1.0 is still valid on that shared trajectory.

## Decision (pick one)

```text
[x] Variant 1 — hidden mean ≥ 0.7, false_revision ≈ 0
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
verdict: capacity_or_seed: hidden direction moved; Gate 03 closed
gate03_closed: yes (Maga 2026-08-20)
next: Gate 04/08 consolidation — belief revision + world model + experience + runtime
      in one loop. Then 6.8B. Pod stays UP. Do not terminate. Do not bump λ.
operator: Maga accepted Variant 1. JSON stays local (experiments/, not in git).
```
