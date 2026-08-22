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

## Order

| # | Job | Machine | Train? |
|---|---|---|---|
| 1 | `python scripts/diagnose_counterfactual_v031.py` | CPU | no |
| 2 | same + `--dataset dataset/mina_6_8b_v03` | H200 or local pack | no |
| 3 | same + `--verdict artifacts/v031/verdict/step1128.json` | H200 | no |
| 4 | Re-run heldout-100 **only if** we keep fork A and want a new official gate on occupied/agent | 1× H200 | **no** |
| 5 | v0.3.2 train | only if 4 still fails after occupied gate | continuation from step1128 |

## If 4 passes

Re-evaluate v0.3.1 as B→maybe A on the **occupied** gate. Weights stay `step1128.mina`. Card stays research until that JSON says A.

## If 4 fails

Then sampler / `L_action` occupied mean / embodiment slices. Still no new DWC.

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
