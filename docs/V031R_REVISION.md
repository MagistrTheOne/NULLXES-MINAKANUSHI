# v0.3.1-R — Revision Trigger Diagnostic

Not v0.4. Not a new general train. Weights stay `step1128.mina`.

```text
v0.3.1 learned the world.
v0.3.1 learned useful memory.
v0.3.1 has action-conditioned futures.

Remaining defect:
when to press "revise belief",
especially delayed embodiment evidence.
```

## What already closed

Occupied heldout-100 (no train): counterfactual existence/diversity PASS on occupied slots. Memory PASS. Heldout ADE 6.78 → 0.845. Variant stays **B** because revision *detection* dropped (0.85 → 0.64). Direction on correction slices is 1.0. The hole is `sensor_delay` detect **0.00** (n=13).

## Cut-points (CPU, no weights — already measured)

1. **sensor_delay forensic.** `delay = 0.15` is `arrival_time − event_time`. `observe()` still writes *current* `xy`. Delayed evidence is a timestamp, not a late picture of an old position.
2. **The trained frame has no mover.** Curriculum stamps corrections at frame 2 *and* `mid`. `training_frame("sensor_delay")` is `length//2`. On length 32 the mover leaves sensor range around frame 6. At frame 16 the only visible body is a static obstacle.
3. **Evidence magnitude even at frame 2.** One-step `≈0.06`, delay-path `≈0.09`, both under `REVISION_MAGNITUDE = 0.25`. A correct tracker would not fire the teacher even on the frame that still sees the mover.
4. **Teacher / calibration.** `should_revise_mask` is only `|belief − evidence| >= 0.25`. Curriculum correction rows are not the teacher. `revision_metrics` scores `n_need==0` as `revision_detected==0.0`.
5. **Constructor path.** While the mover is visible it is consecutive → `tracking`, not `hypothesis_revision`. No `CorrectionEvent`.
6. **Better prediction suppresses the leftover trigger.** At the trained frame the teacher can only fire on the obstacle residual. step128 is far enough to trip 0.25. step1128 tracks the obstacle; detect goes to 0.00.

This is an uncertainty ↔ revision boundary plus a wrong training frame, not a missing counterfactual.

## Commands

CPU:

```text
python scripts/diagnose_revision_v031r.py
python scripts/diagnose_revision_v031r.py --dataset dataset/mina_6_8b_v03
python scripts/diagnose_revision_v031r.py \
  --before-verdict artifacts/v031/verdict/step128.json \
  --after-verdict artifacts/v031/verdict/step1128.json
```

H200 live, `sensor_delay` heldout only (skip the full 100, skip dataset verify scan):

```text
cd /workspace/NULLXES-MINAKANUSHI
git pull --ff-only

python scripts/diagnose_revision_v031r.py \
  --dataset dataset/mina_6_8b_v03 \
  --before-verdict artifacts/v031/verdict/step128.json \
  --after-verdict artifacts/v031/verdict/step1128.json \
  --before /workspace/checkpoints/minakanushi_stage0_step128.mina \
  --after experiments/mina_6_8b_v031/minakanushi_stage0_step1128.mina \
  --out artifacts/v031r/live.json
```

Live dump should print `cut=` per episode. Expect `no_mover_evidence` on the trained frame. If step1128 leftover `max_before_d < 0.25` and step128 is not, the “better prediction suppresses revision” half is also confirmed.

## After the live dump — still no general train

If the hypothesis holds: the next patch is local (teacher that sees delay / tracking-vs-revision split / metric that does not punish empty teacher). Not 1000 more steps. Not new layers. Not v0.4.

If live shows `n_need>0` and `moved≈0`: then the constructor/DWC is swallowing a real correction. That is a different one-line fix, still not a general train.
