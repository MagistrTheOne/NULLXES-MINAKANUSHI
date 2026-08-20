# Training

MINAKANUSHI is not trained with a next-token objective.

## Composite loss

```text
L = λs L_state + λt L_temporal + λf L_future + λu L_uncertainty
  + λc L_causal + λm L_memory + λa L_action + λr L_representation
```

| Term | Property |
|---|---|
| L_state | physical state grounding (xy readout vs simulator) |
| L_temporal | next-step transition |
| L_future | multi-horizon trajectory |
| L_uncertainty | NLL on state-uncertainty channel 6 plus missing-channel calibration |
| L_causal | velocity-structure consistency |
| L_memory | occluded-entity position retention |
| L_action | counterfactual branch separation |
| L_representation | isotropic occupied-latent regularizer |

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

Gate 03 (next, after Gate 02 close): held-out synthetic curriculum plus
adversarial reality checks (belief correction, conflict vs blind average).
Do not redefine the world model as `image → next image` or `state → future_xy`.
Do not start `gpu_train_v01` / `research_v01` as Gate 03.

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
