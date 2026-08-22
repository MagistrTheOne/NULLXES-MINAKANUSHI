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

watch every 50 steps (`experiment.jsonl`):
  loss
  future ADE / FDE
  revision_accuracy
  false_revision
  memory_future_delta
  counterfactual_distance
  heldout_score
```

Do not wait for a pretty curve. After step 1000, compare `capability_before`
vs `capability_after` (`scripts/compare_v031.py`). Not step128 loss vs stepN
loss.

Passport: `artifacts/v031/run_manifest.json`.

If git is dirty, `git_status.json` must say whether **code** is dirty or only
`dataset/` / `artifacts/`. Code must be unambiguous before H200.

Config: `configs/training/mina_6_8b_v03.yaml` (`dataset_split: train`, `steps: 1000`).

Origin pack: `artifacts/v031/baseline/` from `scripts/lock_v031_baseline.py`.
