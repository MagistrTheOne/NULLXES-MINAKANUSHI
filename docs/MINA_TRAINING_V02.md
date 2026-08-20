# MINAKANUSHI 6.8B Status Core v0.2

```text
ARCHITECTURE FREEZE

MINAKANUSHI 6.8B architecture is frozen.

Forbidden:
- changing latent_dim
- changing core_depth
- changing world_slots
- changing memory_slots
- adding layers
- removing layers
- replacing DWC
- adding attention modules
- adding MoE
- adding language head
- identity_loss
- training authority as a neural objective

v0.2 improves training pipeline and data only.
```

Frozen: architecture, SelfModel schema, Authority schema, ActionIntent contract.

Trainable: world prediction, belief update, uncertainty, memory usefulness, future branches, causal understanding.

```text
step64.mina
  → Identity Initialization
  → IdentityBound.mina
  → JsonEpisodeDataset   (dataset/mina_6_8b = SOURCE OF TRUTH)
  → Physics → Agency → Causality → Embodiment
  → resume (same optimizer, identity, RNG, cursor)
  → Status Core v0.2
  → Acceptance Gate
  → Yunmu review
```

HF Minari / D4RL / Open-X are adapters only. See `docs/HF_WORLD_MODEL_ADAPTERS.md`.

## Identity Initialization

Not training. Load checkpoint zip → stamp passport + authority schema → fail-loud validate → save `MINA-6.8B-IdentityBound.mina`. No `identity_loss`. No 6.8B construct required.

```text
python scripts/identity_init.py --checkpoint path/to/step64.mina --out path/to/MINA-6.8B-IdentityBound.mina
```

## JSON curriculum

```text
python scripts/generate_6_8b_curriculum.py --root dataset/mina_6_8b --n 250
python scripts/audit_curriculum.py --root dataset/mina_6_8b --out dataset/mina_6_8b/dataset_report.json
```

`--n 250` → 1000 episodes (250 × 4 phases). JSON blobs stay off git.

## Resume

```text
torchrun --nproc_per_node=1 scripts/train.py \
  --config configs/training/mina_6_8b_v02.yaml \
  --out experiments/mina_6_8b_v02 \
  --resume path/to/MINA-6.8B-IdentityBound.mina
```

`steps` in YAML is this job's step count. Resume starts at `manifest.train.step + 1`.

## Core metrics

World: ADE, FDE, uncertainty calibration  
Belief: revision_accuracy, revision_latency, false_revision  
Memory: memory_future_delta (positive = memory improves next-state error)  
Future: future_diversity, counterfactual_quality  

## v0.1 Status Core ledger (step 64)

PASS (engine/safety): construct, forward, backward, AdamW, `.mina`, `constraint_violation_count==0`, closed loop 1.0, persistence/reacquisition 1.0, ActionIntent only.

Not PASS (learning): JSON not in the loss, no resume, revision accuracy vacuous after step 1, branch coverage ≈ 0, memory_effect_delta is latent L2 not future help.

## Acceptance Gate

```text
python scripts/gate_v02_acceptance.py
```

See `docs/GATE_V02_ACCEPTANCE.md`.

Can MINA: predict world, detect wrong belief, revise, remember, choose different future, respect authority.

If not: do not add layers. Fix data and resume.
