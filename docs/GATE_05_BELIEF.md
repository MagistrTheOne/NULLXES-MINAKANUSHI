# Gate 05 — Belief Engine

**Status:** implemented on `cpu_dev`
**Does not:** instantiate `research_v01`, train at scale, or implement Gates 06–11

Identity hierarchy (unchanged):

```text
NULLXES → MINAKANUSHI → MINA → instance
```

Persona / voice / face stay above cognition. Gate 05 does not touch them.

## Contract

```text
Observation + Previous Belief + Memory + Dynamics → Updated Belief
```

Public object is **Belief**, not a latent bag. `latent_state` remains the
learned carrier. Do not rename DWC internals to `belief_state`.

```text
BeliefSlot
  position_mean, position_std
  velocity_mean, velocity_std
  existence_probability
  last_observation_age
  prediction_confidence
  cause_history          # CorrectionEvent tuple, not RAG
```

`WorldState.as_belief()` is that view. Occupied is slot allocation.
Existence is the soft “I think this is real.”

## Update laws (constructor, not a new net)

- First detect: mean = evidence, std = observation noise, existence high.
- Tracking / revision: `revise_slot` updates means; std shrinks when
  evidence dominates and inflates on conflict.
- Unobserved persist: coast mean by `vel * dt`; inflate `xy_std`; decay
  existence toward retirement. Existence does not snap to 0 until
  `persistence.steps`.
- Memory hints may shift an **unobserved mean** slightly. They never
  50/50 with live evidence.
- After DWC, kinematics copy into belief mean. Residual magnitude enters
  `xy_std` / `pred_confidence`. FutureEngine must not write back.

Invariant from 03A: Future explores possibility; only new evidence plus
constructor/DWC update belief.

## Objective

`L_belief` = Gaussian NLL of GT xy under `(mean, std)` + BCE of existence
vs “was actually present.” `L_state` stays as auxiliary grounding.
`λ_belief` is set on Stage 0 / `cpu_dev` training YAML only.

## Organism order after this gate

```text
06 Memory as Experience
07 Curiosity / Focus Engine
08 Active World Model (Belief_t + Action → Belief_t+1)
09 Autonomous Runtime
10 Embodiment adapters
11 Language interface
```

Leave Gate 04 `FocusState` stubs for Gate 07. Experience is Gate 06.
