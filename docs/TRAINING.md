# Training

MINAKANUSHI is not trained with a next-token objective.

## Composite loss

```text
L = λs L_state + λt L_temporal + λf L_future + λu L_uncertainty
  + λc L_causal + λm L_memory + λa L_action + λr L_representation
  + λb L_belief + λv L_revision
```

| Term | Property |
|---|---|
| L_belief | Gaussian NLL of GT xy under `(mean, std)` plus existence vs was-present |
| L_revision | detection + direction + calibration of belief update vs new evidence; DWC residual is in the path |
| L_state | auxiliary physical grounding (xy readout vs simulator); not the world-model definition |
| L_temporal | next-step transition |
| L_future | multi-horizon trajectory |
| L_uncertainty | NLL on state-uncertainty channel 6 plus missing-channel calibration |
| L_causal | velocity-structure consistency |
| L_memory | occluded-entity position retention |
| L_action | counterfactual branch separation |
| L_representation | isotropic occupied-latent regularizer |

`λ_belief` and `λ_revision` are wired on Stage 0 / `cpu_dev` and Stage A
`gpu_train_v01` training YAMLs. Other stages default to 0 until those gates
own the term. `L_revision` is not `loss += correction_count`. It is the
textbook for "new evidence beats the old hypothesis": detection, direction
toward evidence, and calibration. GATE03 cases
(`hidden_correction`, `conflict`, `reacquisition`, `gone_forever`) are in
the training curriculum, not only OOD.

August 2026 world-model papers (PhyLatent, PSG-JEPA, LeWM) motivate the
physical-grounding, multi-horizon, and counterfactual terms. They do not
define MINAKANUSHI identity. Cosmos 3 / π0.5 token-VLA stacks are not the
runtime.

## Curriculum

| Stage | File | Goal |
|---:|---|---|
| 0 | `configs/training/stage0_validation.yaml` | architecture validation on SyntheticWorld |
| 1 | `configs/training/stage1_world.yaml` | observation → WorldState |
| 2 | `configs/training/stage2_temporal.yaml` | S_t → S_{t+1} and futures |

Gate 03A (belief revision) and Gate 05 (Belief Engine) are implemented on
`cpu_dev`. Do not redefine the world model as `image → next image` or
`state → future_xy`. GPU order: `docs/TRAINING_PLAN.md`. Product target is
**MINA 6.8B** (`docs/MINA_6_8B_TRAINING.md`). Stage A on 6000 BW uses
`gpu_train_v01` as instrument only. Loss decrease is not acceptance.

Later stages (memory stress, OOD uncertainty, strategy ranking, adversarial
constraints, closed-loop, physical integration) are specified but not yet
active YAML. SelfModel / Authority implementation is Gate 04, structured
state only — no identity network. See `docs/GATE_03_PRE_WORLD_MODEL.md`.

## Checkpoint (v3: SAFETENSORS-INSIDE / PICKLE-OUT)

`*.mina` = zip(`manifest.yaml`, `architecture.yaml`, `identity.json`,
`weights/tensors_index.json`, `weights/system-*.safetensors`,
`weights/optimizer.safetensors`?, `weights/runtime.safetensors`?,
`weights/sidecar.json`).

No `*.pt`, no `torch.save`/`torch.load`, no `weights_only=False` on the
`.mina` path. Any legacy pickle entry, hash mismatch, unexpected file, or
shape/dtype mismatch rejects BEFORE model mutation. Old `weights.pt`
checkpoints are invalid (strict, no migrator).

Manifest requires `architecture: MINAKANUSHI` and `organization: NULLXES`.
Load fails on latent_dim mismatch instead of silent reshape.

Hub role: `AutoConfig`/`AutoModel` is a type tag only
(`minakanushi.hub`, never `AutoModelForCausalLM`); research scale
(1B+ params, frozen 6.8B dims, or latent>=4096) refuses construct.
Runtime load is always `load_mina`. The public safetensors mirror
(`model-*.safetensors` + `model.safetensors.index.json`) is Hub vitrine,
not the runtime; verify it with `scripts/test_hf_reload.py`.

## Metrics that count

Loss decrease is not success. Report trajectory error, persistence occupancy
under occlusion, counterfactual separation, hard-constraint reject rate, and
the split revision metrics (`revision_detected`,
`revision_direction_accuracy`, `revision_magnitude_error`,
`revision_latency`, `false_revision_rate`). `belief_revision_accuracy` is
the direction score, not a silent zero.
