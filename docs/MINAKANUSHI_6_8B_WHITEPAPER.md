# NULLXES MINAKANUSHI 6.8B

**Technical report · architecture generation 1 · research checkpoint**

```text
Organization:     NULLXES
Architecture:     MINAKANUSHI
Short name:       MINA
System class:     adaptive situational intelligence
Native runtime:   nullxes
Parameters:       6 799 130 646
Freeze:           7aba976
Canonical format: *.mina
Public mirror:    safetensors (weights only)
```

Status of the latest published weights (`minakanushi_stage0_step1128.mina`):

```text
Research checkpoint
Training cycle: v0.3.1
Accepted: NO
Capability verdict: pending / H200 heldout-100 protocol
Not a language model
Action output: ActionIntent
PWM: false
```

This document is a measurement report. It is not a claim of general intelligence.

---

## 1. Problem

MINAKANUSHI infers the state of a changing physical environment from incomplete observations, represents uncertainty, predicts multiple futures, and selects an admissible strategy under human constraints.

The primitive is a closed loop:

```text
observation → perception → world state → temporal state → uncertainty
→ predicted futures → situation → strategies → constraints → ActionIntent
→ new observation
```

Language is a modality, not the architecture. The token is not the unit of cognition.

---

## 2. Identity

Identity is checkpoint metadata, not a prompt and not a trained “I am MINA” head.

```yaml
architecture: MINAKANUSHI
organization: NULLXES
system_class: adaptive_situational_intelligence
architecture_generation: 1
native_runtime: nullxes
architecture_version: "0.1"
```

SelfModel is structured passport state (identity, capabilities, embodiment, authority, runtime). Authority gates action permission. It does not erase world understanding and does not bypass hard constraints. `policy_enabled=false` is cognition on, autonomous selection off.

Belief `B_t` is a probabilistic world state, not a hidden embedding only.

---

## 3. Architecture

Frozen research profile `minakanushi_6_8b`:

| Field | Value |
|---|---|
| `latent_dim` / `state_dim` / `memory_dim` | 4096 |
| Dynamic World Core depth | 32 |
| World slots / memory slots | 512 / 1024 |
| Uncertainty channels | 8 |
| Future branches | 3 |
| Cognition budget / convergence | 4 / 0.02 |
| Prediction horizons | 1 / 4 / 8 |
| `dt` | 0.1 s |

Native data unit: **MinaUnit** (source, time, optional space, embedding, confidence, persistence, causal parents). Position is **NullxesPositionField** (sequence, time, space, episode, memory age, source) — not token index.

Subsystems: Perception Bridge → State Constructor → Dynamic World Core → Memory Engine (working / episodic / semantic) → Uncertainty Engine → Situation Core → Future Engine → Strategy Engine → Constraint Kernel (MCK) → Action Policy.

Hard constraints cannot be overridden by confidence, predicted value, or memory. The core emits **ActionIntent**. It does not emit motor PWM.

---

## 4. Data

Source of truth: NULLXES SyntheticWorld. Not a hosted video corpus. Not an external foundation-model dump.

v0.3.1 pack `dataset/mina_6_8b_v03` / Hub `MagistrTheOne/mina-6.8b-v03`:

```text
episodes:              1000
train / heldout:       900 / 100
physics / agency:      32 frames
causality / embodiment: 64 frames
split rule:            episode_index % 10 == 9 → heldout
pwm:                   false
marker:                .READY_V031
```

H200 verifies the pack. It does not regenerate or repair splits.

---

## 5. Training contract

Allowed: dataset, sampler, optimizer schedule, checkpointing, metrics.

Forbidden: changing latent/depth/slots, replacing DWC, new heads, language adapter, RGB, `identity_loss`, MoE, CausalLM export, constructing 6.8B on CPU / consumer GPUs.

Published checkpoints:

| Cycle | File | Machine | Role |
|---|---|---|---|
| v0.1 | `minakanushi_stage0_step64.mina` | 1× H200 | Status Core construct / loop |
| v0.2 | `minakanushi_stage0_step128.mina` | 1× B300 | IdentityBound + JSON resume |
| v0.3.1 | `minakanushi_stage0_step1128.mina` | 1× H200 | 1000-step research run, **not accepted** |

Every published model after v0.3.1 ships `*.mina` (runtime, optimizer, identity) and a safetensors weight mirror. Load path remains `load_mina`. Hub `AutoModel` is a type tag and refuses research-scale construct.

---

## 6. Evaluation protocol

A lower train loss is not a new ability. Single-episode eval during training is a monitor, not the ledger.

H200 verdict (`scripts/gate_v031_h200_verdict.py`) compares step128 vs step1128 on **all 100 heldout episodes**:

1. Aggregates (mean, median, p90, worst-10) for ADE, FDE, uncertainty, revision, false revision, revision direction — overall and by phase.
2. Memory ON vs OFF ADE/FDE on the same episodes and seed. Memory scenarios: occlusion, delayed, sensor_delay, reacquisition, hidden_object. Gate: ADE(on) < ADE(off). Latent L2 is not this gate.
3. Counterfactual intervention: one world state, WAIT vs MOVE_TO. Terminal, trajectory, and relation deltas.
4. Revision slices: hidden_correction L1–L3, conflict, unexpected_stop, gone_forever. Detection, direction, false revision, recovery latency.
5. Action-path trace (20 states): `||action_WAIT − action_MOVE_TO||` vs `||F_WAIT − F_MOVE_TO||`.

Variants:

| | Meaning |
|---|---|
| **A** | Heldout ADE down, revision up, memory helps, false revision stays low, futures separate |
| **B** | Train / some ADE moved; memory, direction, or action conditioning did not |
| **C** | Revision honesty broke or false revision rose |

v0.4 (geometry / long horizon) starts only after A.

---

## 7. Measured so far

### v0.1 · step 64 · H200

Persistence / reacquisition 1.0. Hard-constraint violations 0. Future ADE / FDE 3.42 / 0.81. JSON was not yet in the loss. Not an intelligence pass.

### v0.2 · step 128 · B300

Future ADE / FDE 2.05 / 0.68. Revision accuracy 0.0. Branch coverage 0.0. False revision 0.0. Resume preserved identity and optimizer. Not an intelligence pass.

### v0.3.1 · step 1128 · H200

Train-eval monitor (one heldout episode per 50 steps) is **not** the verdict. Hint only: heldout ADE 1.61 → 0.15 → bounce 0.44; revision detection often live; `cf ≈ 0.0008`; action term ≈ 0.499; false revision appeared after a step-1080 spike.

Official numbers: `artifacts/v031/verdict/compare.json` after the H200 heldout-100 job.

---

## 8. Limits

- Synthetic arena. Not locomotion. Not vision-foundation trained.
- Counterfactual diversity on one geometry is not a world model of many geometries.
- Safetensors is a mirror. Optimizer and identity stay in `*.mina`.
- Redistribution as another model family, or as an LLM wrapper with this name, is not granted.

Hub: [MagistrTheOne/MINAKANUSHI-6.8B](https://huggingface.co/MagistrTheOne/MINAKANUSHI-6.8B)  
Collection: [ASI (WorldModel)](https://huggingface.co/collections/MagistrTheOne/asi-worldmodel-6a89f942152bb18dd68c144b)  
Dataset: [mina-6.8b-v03](https://huggingface.co/datasets/MagistrTheOne/mina-6.8b-v03)

NULLXES MINAKANUSHI Research License.

```text
I WILL SURVIVE.
NULLXES.
```
