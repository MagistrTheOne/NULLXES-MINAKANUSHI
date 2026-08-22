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

First GPU for this cycle: **1× H200 SXM 141 GB** (same class as v0.1 step64). Download
`minakanushi_stage0_step128.mina` from Hugging Face onto that pod. Do not raise B300
just to hash a zip. RTX 2080 does not see the 27GB file.

On the laptop / RTX 2080, before the pod:

```text
python scripts/prepare_v031_dataset.py --root dataset/mina_6_8b_v03 --n 250 --seed 11
python scripts/check_freeze.py
python scripts/gate_v031_acceptance.py --dataset dataset/mina_6_8b_v03 --split heldout
python scripts/register_hf_architecture.py
```

`prepare_v031_dataset.py` is the only production builder (`--n` default 250, `--profile v031`).
`--profile cpu_dev` is toys. H200 refuses a pack without `.READY_V031` (and refuses cpu_dev markers).
Copy `dataset/mina_6_8b_v03/` to the pod. Do not generate on H200.

On H200 after `hf download` and the copied pack:

```text
python scripts/verify_v031_dataset.py --root dataset/mina_6_8b_v03
python scripts/lock_v031_baseline.py --mina minakanushi_stage0_step128.mina --require-mina --out artifacts/v031/baseline
python scripts/check_freeze.py --checkpoint minakanushi_stage0_step128.mina
python scripts/export_safetensors.py --mina minakanushi_stage0_step128.mina --out MINAKANUSHI-6.8B
python scripts/test_hf_reload.py --path MINAKANUSHI-6.8B
```

`--require-mina` also requires a verified v0.3.1 dataset pack. `verify` is read-only.

If git is dirty, `git_status.json` must say whether **code** is dirty or only
`dataset/` / `artifacts/`. Code must be unambiguous before H200.

Config: `configs/training/mina_6_8b_v03.yaml` (`dataset_split: train`, `steps: 1000`).

Origin pack: `artifacts/v031/baseline/` from `scripts/lock_v031_baseline.py`.
