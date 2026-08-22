# MINAKANUSHI Training Contract v0.3

Pre-training lock for the H200 experiment. Architecture freeze `7aba976` stays.

This run is not “make MINA smarter.” It is a clean before/after measurement:
what changed after learning, with the generator, split, and checkpoint pinned.

```text
ALLOWED
  dataset changes
  sampler changes
  optimizer schedule
  checkpointing
  metrics

FORBIDDEN
  DWC changes
  latent changes
  slots changes
  new heads
  language adapter
  RGB input
```

Also forbidden on this job: identity_loss, MoE, extra layers, CausalLM export,
constructing `minakanushi_6_8b` on CPU / RTX PRO 6000 / 1× H100 80GB.

Canonical weights remain `*.mina`. safetensors is a Hub mirror only.

## Phase 1 on H200

```text
1000 steps
then STOP

watch:
  loss
  future ADE/FDE
  revision
  memory_future_delta
  counterfactual
```

Do not wait for a pretty curve. After step 1000, run retention (Gate A) and
held-out (Gate B). Update `docs/MINA_CAPABILITY_LEDGER.md` only from counted
measurements (`n=`, ADE on/off), never from `loss=`.

Config: `configs/training/mina_6_8b_v03.yaml` (`dataset_split: train`, `steps: 1000`).

Origin pack: `artifacts/v031/baseline/` from `scripts/lock_v031_baseline.py`.
