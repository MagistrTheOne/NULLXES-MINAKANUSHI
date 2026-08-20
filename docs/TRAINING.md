# Training

MINAKANUSHI is not trained with a next-token objective.

## Composite loss

```text
L = λs L_state + λt L_temporal + λf L_future + λu L_uncertainty
  + λc L_causal + λm L_memory + λa L_action + λr L_representation
  + λb L_belief
```

| Term | Property |
|---|---|
| L_belief | Gaussian NLL of GT xy under `(mean, std)` plus existence vs was-present |
| L_state | auxiliary physical grounding (xy readout vs simulator); not the world-model definition |
| L_temporal | next-step transition |
| L_future | multi-horizon trajectory |
| L_uncertainty | NLL on state-uncertainty channel 6 plus missing-channel calibration |
| L_causal | velocity-structure consistency |
| L_memory | occluded-entity position retention |
| L_action | counterfactual branch separation |
| L_representation | isotropic occupied-latent regularizer |

`λ_belief` is wired on Stage 0 / `cpu_dev` training YAMLs only. Other stages
default to 0 until those gates own the term.

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
`state → future_xy`. GPU order: `docs/TRAINING_PLAN.md`. Stage A is
`gpu_train_v01` on RTX PRO 6000 Blackwell; `research_v01` is not this
budget. Loss decrease is not acceptance.

Later stages (memory stress, OOD uncertainty, strategy ranking, adversarial
constraints, closed-loop, physical integration) are specified but not yet
active YAML. SelfModel / Authority implementation is Gate 04, structured
state only — no identity network. See `docs/GATE_03_PRE_WORLD_MODEL.md`.

## Checkpoint

`*.mina` = zip(`manifest.yaml`, `weights.pt`, `identity.json`).

Manifest requires `architecture: MINAKANUSHI` and `organization: NULLXES`.
Load fails on latent_dim mismatch instead of silent reshape.

## Metrics that count

Loss decrease is not success. Report trajectory error, persistence occupancy
under occlusion, counterfactual separation, and hard-constraint reject rate.
