# v0.3.2 diagnostic-fix — one page

Not a new training cycle. Not new layers. Not another 1000 steps.

```text
step1128 already predicts the world.
Acceptance died in one number.
```

## Verdict we are not reopening

B. Heldout ADE 6.78 → 0.845. Memory PASS. Direction 0.41 → 0.64. Official `cf` diversity FAIL.

## Fork A is already decided (CPU, no GPU)

Official score:

```text
cf = mean_i ||Δxy_i||     over world_slots = 512
```

Identity, confirmed by unit test and the H200 ledger:

```text
0.40054  (Future Engine Frobenius, n=20)
÷ 512
= 0.000782
≈ 0.000786 official cf
```

Recovered agent std = `3.4e-6 * 512 ≈ 0.0017 > 1e-4` → occupied/agent diversity would pass.

`L_action` uses the same all-slot mean. Margin 0.25 needs agent Δ ≈ 128. That loss never saturates. Do not “fix” it with more steps.

## Occupied / agent gate (frozen before the next heldout-100)

```text
cf_all_slots  = mean_i ||Δxy_i||     over all world_slots   ← log only
cf_occupied   = mean   ||Δxy_i||     over occupied slots    ← official gate
cf_agent      =        ||Δxy_agent||                        ← action-self check

existence PASS:  max(cf_occupied) > 1e-4
diversity PASS:  std(cf_occupied) > 1e-4
```

`cf_all_slots` is the v0.3.1 artifact. It must not decide acceptance.

Constants: `CF_EXISTENCE_MIN` / `CF_DIVERSITY_MIN` in `minakanushi/training/v031_verdict.py`.

## H200 verification — no optimizer

Do not run until this commit is on the pod. Then, from `/workspace/NULLXES-MINAKANUSHI`:

```text
git pull --ff-only

python scripts/diagnose_counterfactual_v031.py \
  --out artifacts/v032/diagnostic_metric.json

python scripts/diagnose_counterfactual_v031.py \
  --verdict artifacts/v031/verdict/step1128.json \
  --out artifacts/v032/diagnostic_embodiment.json

python scripts/diagnose_counterfactual_v031.py \
  --dataset dataset/mina_6_8b_v03 \
  --first-step 129 --last-step 1128 --resume-start 129 \
  --out artifacts/v032/diagnostic_sampler.json

python scripts/gate_v031_h200_verdict.py \
  --before /workspace/checkpoints/minakanushi_stage0_step128.mina \
  --after experiments/mina_6_8b_v031/minakanushi_stage0_step1128.mina \
  --dataset dataset/mina_6_8b_v03 \
  --out artifacts/v031/verdict_occupied
```

Sampler RNG is global step `129..1128`. Mode is job `1..1000`. Replay of `1..1000` is the wrong experiment.

## GO / NO-GO after occupied heldout-100

GO without train if occupied diversity PASS, memory PASS, heldout ADE still down, revision direction ≥ 0.2, false revision ≤ 0.1.

Then v0.3.1 is **B caused by metric artifact** (or A if the frozen gate closes).

NO-GO on acceptance if `cf_occupied` / `cf_agent` still FAIL. Only then targeted v0.3.2 train from step1128.

Weights stay `step1128.mina`. Card stays research until the occupied ledger says A.

## Embodiment (do not touch global objective)

```text
hidden_correction L1–L3 / conflict : detect 1.0 direction 1.0
embodiment mean detect 0.24
gone_forever n=3 detect 0.33 false_rev 0.67
```

First: more heldout support for `gone_forever`. Check teacher vs persistence. Split embodiment into `sensor_delay` vs `agent_move`.

## Forbidden

```text
no 1000 more steps “to push cf”
no new layers
no Yunmu
no v0.4 geometry
no changing latent / depth / slots
```
