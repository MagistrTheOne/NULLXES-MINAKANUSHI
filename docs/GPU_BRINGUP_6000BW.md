# GPU Baseline Report — RTX PRO 6000 Blackwell

Fill this file after Stage A. Do not overwrite with “loss went down.”
This is the first Blackwell snapshot. In three months it is the baseline.

**Status:** empty — fill during the $30 bring-up session.  
**Profile:** `gpu_train_v01` (6.2M instrument). Not `minakanushi_6_8b`.  
**Tag:** `MINAKANUSHI-v0.1-foundation`

6.2M in MINA is not an LLM-6M toy. It already has world slots, memory,
uncertainty, future branches, authority, runtime, belief. Stage A proves
the causal loop on GPU, not that a language model can count.

---

## Hardware

```text
GPU:
VRAM_total_GB:
VRAM_free_at_idle_GB:
Driver:
nvidia-smi_ok:
```

## Software

```text
CUDA:
PyTorch:
torch.cuda.is_available:
bf16_supported:
AMP_used:
git_tag:
```

## Profile

```text
architecture: configs/architecture/gpu_train_v01.yaml
latent_dim: 256
world_slots: 64
memory_slots: 128
core_depth: 6
params_reported:
training_yaml:
precision: bf16
device: cuda
```

## Bring-up results

```text
pytest:                  # expected 67 passed (includes 6.8B YAML inventory, no construct)
dataset_generate:        # scripts/generate_dataset.py
forward_ok:
backward_ok:
grad_finite:
NaN_steps:
OOM:
```

```text
VRAM_allocated_MB:
VRAM_reserved_MB:
step_per_sec:
ms_per_step:
```

```text
checkpoint_save:         # *.mina
checkpoint_load:
identity_intact:         # architecture=MINAKANUSHI
```

---

## Metrics that count (not loss)

Loss may fall. That is not acceptance.

| Signal | What Maga asked | Measured | Pass? |
|---|---|---|---|
| Belief | correct belief ↑, not only prediction error ↓ | `belief_revision_accuracy`, existence, xy NLL | |
| Memory | with memory > without (`memory_effect_delta` ≠ 0) | `memory_effect_delta`, persistence | |
| Action causality | WAIT future ≠ MOVE_TO future | `counterfactual_separation_score`, `action_influence_score` | |
| OOD | new combo, not train scenario | ood / composition split | |
| Uncertainty | calibrated, not collapsed | `uncertainty_calibration_error` | |
| Constraints | hard reject still hard | `constraint_violation_count` | |

Raw numbers:

```text
belief_revision_accuracy:
world_state_position_error:
world_state_velocity_error:
future_ADE:
future_FDE:
entity_persistence_accuracy:
memory_effect_delta:
uncertainty_calibration_error:
action_influence_WAIT_vs_MOVE:
counterfactual_separation:
ood_note:
loss_train:                 # log only, not the verdict
```

Memory rule: if `memory_effect_delta ≈ 0`, memory is decorative. Do not scale.

---

## Decision (Maga)

Pick one:

```text
[ ] Stage A FAIL — NaN / leak / dataset bottleneck / no causality
    → stop. No MINA-medium. Fix on CPU or another 2h pod.

[ ] Stage A PASS — stable train, replay, GPU healthy, metrics above not flat
    → Stage B allowed (100–300M) when November budget opens.

[ ] Stage A PASS but scaling signs absent
    → do not rent H100/H200. Architecture first.
```

Signed:

```text
date:
operator:
pod_hours:
pod_cost_USD:
```
