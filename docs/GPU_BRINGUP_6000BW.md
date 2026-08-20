# GPU Baseline Report — RTX PRO 6000 Blackwell

Fill this file after Stage A. Do not overwrite with “loss went down.”
This is the first Blackwell snapshot. In three months it is the baseline.

**Status:** filled 2026-08-20. Stack bring-up on `gpu_train_v01`.  
**Profile:** `gpu_train_v01` (6.2M instrument). Not `minakanushi_6_8b`.  
**Tag:** `MINAKANUSHI-v0.1-foundation` (`9e4e3d7`)

6.2M in MINA is not an LLM-6M toy. It already has world slots, memory,
uncertainty, future branches, authority, runtime, belief. Stage A proves
the causal loop on GPU, not that a language model can count.

---

## Hardware

```text
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
VRAM_total_GB: 95.59 (97887 MiB)
VRAM_free_at_idle_GB: 94.97 (97250 MiB)
Driver: 580.126.20
nvidia-smi_ok: yes
vCPU: 48 (Xeon 6952P)
RAM_GB: 188
pod: gn3eqwxuht23qs
image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
spend_rate_USD_per_hr: 2.40
```

## Software

```text
CUDA: 12.8 (nvidia-smi reports 13.0 driver CUDA)
PyTorch: 2.8.0+cu128
torch.cuda.is_available: True
bf16_supported: True
AMP_used: True (torch.autocast cuda/bf16; weights stay fp32)
git_tag: MINAKANUSHI-v0.1-foundation
git_sha: 9e4e3d7
```

Torch 2.8 default `torch.load(weights_only=True)` broke `*.mina` restore.
Patched for native checkpoints (NULLXES format, not untrusted pickles).
On `MINAKANUSHI-revision-gate` (`0ea5062`) as `weights_only=False`.

## Profile

```text
architecture: configs/architecture/gpu_train_v01.yaml
latent_dim: 256
world_slots: 64
memory_slots: 128
core_depth: 6
params_reported: 6_241_302  (numel == formula inventory)
training_yaml: configs/training/stage_a_gpu_train_v01.yaml
precision: bf16 (AMP)
device: cuda
steps: 200
```

## Bring-up results

```text
pytest:                  67 passed (after weights_only=False)
dataset_generate:        ok (train/validation/composition/ood/counterfactual, n=8 length=12)
forward_ok:              yes
backward_ok:             yes
grad_finite:             yes
NaN_steps:               0
OOM:                     no
6.8B_constructed:        no
```

```text
VRAM_allocated_MB: 24.0   (construct gpu_train_v01)
VRAM_reserved_MB:  44.0
construct_s:       0.199
step_per_sec:      ~7     (fwd 0.07s + bwd 0.07s steady)
ms_per_step:       ~140
```

```text
checkpoint_save:         experiments/stage_a/minakanushi_stage0_step200.mina (66M)
checkpoint_load:         probe.mina identity ok
identity_intact:         architecture=MINAKANUSHI
```

---

## Metrics that count (not loss)

Loss may fall. That is not acceptance.

| Signal | What Maga asked | Measured | Pass? |
|---|---|---|---|
| Belief | correct belief ↑, not only prediction error ↓ | `belief_revision_accuracy` = **0.0** at steps 1/50/100/150/200 | **no** |
| Memory | with memory > without (`memory_effect_delta` ≠ 0) | 824 → 2.82 → 1.51 → 39.3 → **1.60** | **yes** (non-zero) |
| Action causality | WAIT future ≠ MOVE_TO future | `branch_diversity` 0.0006 → **0.21**; no dedicated WAIT≠MOVE column in this log | **partial** |
| OOD | new combo, not train scenario | dataset splits written; not evaluated this run | not measured |
| Uncertainty | calibrated, not collapsed | `uncertainty_calibration_error` 0.405 → **0.303** | weak / log only |
| Constraints | hard reject still hard | `constraint_violation_count` = **0** | **yes** (this episode set) |

Raw numbers (step 200):

```text
belief_revision_accuracy: 0.0
world_state_position_error: 0.0180
world_state_velocity_error: 0.0491
future_ADE: 0.699
future_FDE: 0.227
entity_persistence_accuracy: 1.0
memory_effect_delta: 1.601
uncertainty_calibration_error: 0.303
action_influence_WAIT_vs_MOVE: not logged this run
counterfactual_separation: branch_diversity 0.210
ood_note: splits generated, not scored
loss_train: step1 78.68 → step200 -1.93   # log only, not the verdict
```

Memory rule: if `memory_effect_delta ≈ 0`, memory is decorative. Do not scale.
Here it is not decorative.

Belief rule: accuracy stayed 0 while position error fell. That is **not**
belief-revision success. Do not treat this as a 6.8B training green light.

---

## Decision (Maga)

Pick one:

```text
[ ] Stage A FAIL — NaN / leak / dataset bottleneck / no causality
    → stop. No MINA-medium. Fix on CPU or another 2h pod.

[x] Stage A PASS — stable train, replay, GPU healthy
    → stack instrument works on Blackwell (CUDA, bf16 AMP, *.mina).
    Belief metric did not move. Do not rent H200 for 6.8B on this evidence.

[x] Stage A PASS but scaling signs absent (belief)
    → do not rent H100/H200 for 6.8B train until belief_revision_accuracy
      is a real signal. Architecture / metric / data first.
```

**STOP THE POD.** Hourly rate is $2.40. Stage A is done. Do not leave 8h on.

Signed:

```text
date: 2026-08-20
operator: Maga / Cursor
pod_hours: ~0.3 compute after SSH (pod wall-clock longer — terminate now)
pod_cost_USD: ~$0.70 compute + disk until stopped
```
