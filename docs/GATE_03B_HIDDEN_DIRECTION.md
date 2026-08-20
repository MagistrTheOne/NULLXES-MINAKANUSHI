# Gate 03B — Hidden correction direction

**Status:** diagnostic subtask of Gate 03. Not a new architecture gate.
**Parent:** `docs/GATE_03_REVISION_VALIDATION.md`
**Does not change DWC. Does not change λ. Does not construct 6.8B.**

Gate 03 already showed:

```text
GPU stack        ✅
Revision signal  ✅
Dataset path     ✅
Metric           ✅
False revision   ✅
Reacquisition    ✅  direction 1.0
Conflict         ✅  direction 1.0
Hidden correction ⚠️ direction 0.5
```

0.5 is a diagnosis, not a red fail.

## What 0.5 means

MINA can say the old hypothesis is wrong. On `hidden_correction` it does
not yet fully commit to the new one.

```text
old belief     x ≈ 4.53
evidence       x ≈ 2.44
constructor    "revise"
DWC            "revise, but smooth"
```

Conflict has a hard A ≠ B. Reacquisition has disappear → appear.
Hidden correction has a physical prior: it was moving, so it probably
still is. DWC was trained to keep dynamic continuity. That is the
balance under test:

```text
physics prior  vs  new evidence
```

## Question

Is 0.5 a `cpu_dev` / single-seed artifact, or a stable conflict on
`gpu_train_v01`?

```text
hidden direction  0.5 → 0.7+     capacity / seed. Close Gate 03.
hidden direction  stuck at ~0.5  prior vs evidence. Do not bump λ yet.
```

If stuck, read term ratios in the JSON (`revision` vs `state` vs `future`).

```text
A  L_revision too weak relative to L_state / L_future
   → still do not change λ on the first 03B pass; record the ratio.

B  DWC keeps velocity prior too well
   → later: causal correction ("why the forecast was wrong"),
     not a new slot and not a bigger model.
```

## Run

Identity + `λ_revision = 1.0` are required. Eval only. Not a full train.

CPU smoke (`cpu_dev`, small N):

```text
python scripts/gate03b_hidden_direction.py --n 8
```

Blackwell (`gpu_train_v01`, Maga's N):

```text
python scripts/gate03b_hidden_direction.py \
  --training configs/training/stage_a_gpu_train_v01.yaml \
  --n 1000 \
  --out experiments/gate03b
```

Optional `--checkpoint path.mina` after a post-fix train. Stage A
checkpoints from before `0ea5062` are a different experiment.

Do not construct `minakanushi_6_8b`. Stop the pod after the JSON.
Fill `docs/GATE_03B_HIDDEN_DIRECTION_REPORT.md` from that JSON.

## NumPy warning

`NumPy 2.2.6 _ARRAY_API not found` on Windows/torch is environment, not
Gate 03. It did not change the 0.5. On the pod prefer the image's torch
bundle; pin `numpy<2` only if `nvidia-smi` / pytest are otherwise clean
and the warning becomes a crash.

## Closed until 03B answers

```text
model scale
MINA 6.8B
H200
architecture edit
new world slots
λ_revision retune
```

## Route

```text
3d8012e
  ↓
Blackwell Gate 03B  (hidden × N, conflict × N, reacquisition × N)
  ↓
hidden direction rises  →  close Gate 03
hidden stuck            →  DWC evidence-weight analysis (A/B)
  ↓
Gate 08 consolidation
  ↓
6.8B
```
