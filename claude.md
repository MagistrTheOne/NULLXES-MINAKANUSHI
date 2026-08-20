# NULLXES MINAKANUSHI

## Architecture & Production Specification v0.1

**Organization:** NULLXES
**System:** MINAKANUSHI
**Class:** Adaptive Situational Intelligence / Physical Intelligence Foundation Architecture
**Status:** Architecture Lock Candidate
**Implementation language:** Python + C++/CUDA where required
**Target:** research runtime → simulator → controlled physical systems

---

# 0. PRIME DIRECTIVE

You are implementing a completely new machine-learning architecture named:

**NULLXES MINAKANUSHI**

MINAKANUSHI is not a wrapper around an existing language model.

MINAKANUSHI is not a fine-tuned chatbot.

MINAKANUSHI is not an agent framework around another foundation model.

MINAKANUSHI is not an inference frontend for another model.

MINAKANUSHI is an adaptive situational intelligence architecture for autonomous physical systems.

Its primary computational problem is:

> infer the current state of a changing physical environment from incomplete observations, estimate uncertainty, predict possible future states, evaluate admissible strategies, and select actions within explicit human-defined constraints.

The implementation MUST NOT propose replacing the architecture with:

* Qwen
* Llama
* Mistral
* Gemma
* DeepSeek
* GPT
* Claude
* existing agent frameworks
* external LLM APIs
* hosted foundation models

Existing libraries may be used as low-level engineering infrastructure for tensors, kernels, serialization, testing, telemetry and hardware access.

They MUST NOT define the cognitive architecture.

The architecture defined in this specification is authoritative.

---

# 1. SYSTEM IDENTITY

The canonical identity is:

```text
NULLXES MINAKANUSHI
```

Manufacturer / research organization:

```text
NULLXES
```

Architecture family:

```text
MINAKANUSHI
```

MINAKANUSHI must possess explicit architectural identity metadata.

Identity must not depend on:

* system prompts;
* chat templates;
* inference prompts;
* hidden instructions;
* frontend text;
* external API metadata.

Canonical model metadata:

```yaml
architecture: MINAKANUSHI
organization: NULLXES
system_class: adaptive_situational_intelligence
architecture_generation: 1
native_runtime: nullxes
```

These values are part of checkpoint and runtime metadata.

---

# 2. FUNDAMENTAL COMPUTATIONAL MODEL

MINAKANUSHI operates on:

```text
OBSERVATION
     ↓
PERCEPTION
     ↓
WORLD STATE
     ↓
TEMPORAL STATE
     ↓
UNCERTAINTY
     ↓
PREDICTED FUTURES
     ↓
SITUATION
     ↓
STRATEGIES
     ↓
CONSTRAINTS
     ↓
ACTION POLICY
     ↓
ACTION
     ↓
NEW OBSERVATION
```

This loop is the fundamental primitive.

Language generation is NOT the fundamental primitive.

---

# 3. MATHEMATICAL STATE

At physical time `t`, define observations:

[
O_t = {o_t^1,o_t^2,...,o_t^n}
]

where observations may originate from:

```text
vision
telemetry
IMU
position
velocity
environmental sensors
system state
operator input
mission context
internal memory
```

MINAKANUSHI constructs latent world state:

[
S_t = F_\theta(O_{\le t}, M_{t-1})
]

State MUST NOT represent only the current observation.

It represents the model's current belief about reality.

Define:

[
S_t =
(
E_t,
R_t,
A_t,
C_t,
T_t,
U_t
)
]

where:

```text
E_t = entities
R_t = relations
A_t = agent/self state
C_t = environmental context
T_t = temporal state
U_t = uncertainty
```

Memory:

[
M_t = G_\theta(M_{t-1},S_t,O_t)
]

Future state prediction:

[
\hat S_{t+\Delta}^{(k)}
=======================

P_\theta(S_t,M_t,a^{(k)})
]

for candidate strategy/action trajectory `k`.

Policy selection:

[
\pi^*
=====

\arg\max_{\pi \in \Pi_{allowed}}
V(S_t,M_t,\hat S,\pi,U_t)
]

subject to:

[
C_{human}(S_t,\pi)=true
]

No strategy violating hard human constraints may enter the executable action set.

---

# 4. MINAKANUSHI NATIVE DATA UNIT

Do NOT make the token the universal unit of cognition.

Introduce:

```text
MinaUnit
```

Canonical structure:

```python
MinaUnit:
    source_type
    source_id

    timestamp
    sequence_index

    spatial_frame
    spatial_position

    semantic_embedding

    confidence
    uncertainty

    persistence

    entity_reference
    relation_reference

    causal_parent_ids

    metadata
```

A MinaUnit represents a temporally and optionally spatially grounded piece of information.

Examples:

```text
visual feature
detected object
velocity measurement
operator instruction
memory event
system fault
environmental observation
predicted event
language concept
```

Text tokens may eventually be converted into MinaUnits.

They are not privileged over physical observations.

---

# 5. NATIVE POSITION SYSTEM

MINAKANUSHI MUST NOT define position solely as:

```text
token index = 0,1,2,3...
```

Native position is multidimensional.

Define:

[
P_i =
(
p_{seq},
p_{time},
p_{space},
p_{episode},
p_{memory},
p_{source}
)
]

## 5.1 Sequence position

Relative ordering inside one observation stream.

## 5.2 Physical time

Actual observation time.

Store internally using monotonically ordered high-resolution timestamps.

The architecture must understand:

```text
observation time
processing time
event time
prediction horizon
```

as different concepts.

## 5.3 Spatial position

Optional coordinate relative to a declared reference frame.

Never assume every MinaUnit has spatial coordinates.

## 5.4 Episode position

Position inside the current continuous interaction episode.

## 5.5 Memory age

[
age_i=t_{current}-t_i
]

Memory age participates in relevance calculation.

## 5.6 Source position

Identifies the originating sensor/modality/stream without conflating it with semantics.

---

# 6. NULLXES POSITION FIELD — NPF

Implement a native position representation named:

```text
NullxesPositionField
NPF
```

NPF maps multidimensional position into latent positional state:

[
NPF(P_i) \rightarrow z_i^p
]

The first implementation MUST contain independently testable encoders for:

```text
sequence
time
space
episode
memory_age
source
```

Their outputs are fused through a trainable position mixer.

Required interface:

```python
class NullxesPositionField(nn.Module):

    def forward(
        self,
        sequence_position,
        timestamp,
        spatial_position,
        episode_position,
        memory_age,
        source_id,
    ) -> PositionState:
        ...
```

PositionState:

```python
PositionState:
    embedding
    temporal_embedding
    spatial_embedding
    episode_embedding
    memory_embedding
    source_embedding
```

Do not alias this implementation to another architecture's positional system.

---

# 7. CORE LATENT DIMENSION

Initial research configuration:

```yaml
latent_dim: 2048
state_dim: 2048
memory_dim: 2048

world_slots: 512
memory_slots: 1024

core_depth: 24

prediction_horizons:
  - immediate
  - short
  - medium

uncertainty_channels: 8
```

All values MUST be configurable.

No model logic may contain hard-coded assumptions about these dimensions.

---

# 8. ARCHITECTURE

The primary architecture is:

```text
                    SENSOR / DATA SPACE
                           │
                           ▼
                ┌────────────────────┐
                │ PERCEPTION BRIDGE  │
                └─────────┬──────────┘
                          ▼
                    MINA UNITS
                          │
                          ▼
                ┌────────────────────┐
                │ NULLXES POSITION   │
                │ FIELD — NPF        │
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │ STATE CONSTRUCTOR  │
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │ DYNAMIC WORLD CORE │
                └─────────┬──────────┘
                          │
               ┌──────────┴───────────┐
               ▼                      ▼
      ┌────────────────┐     ┌────────────────┐
      │ MEMORY ENGINE  │     │ UNCERTAINTY    │
      │                │     │ ENGINE         │
      └───────┬────────┘     └────────┬───────┘
              └───────────┬───────────┘
                          ▼
                ┌────────────────────┐
                │ SITUATION CORE     │
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │ FUTURE ENGINE      │
                └─────────┬──────────┘
                          ▼
                  FUTURE BRANCHES
                          │
                          ▼
                ┌────────────────────┐
                │ STRATEGY ENGINE    │
                └─────────┬──────────┘
                          ▼
                 CANDIDATE STRATEGIES
                          │
                          ▼
                ┌────────────────────┐
                │ CONSTRAINT KERNEL  │
                └─────────┬──────────┘
                          ▼
                  ALLOWED STRATEGIES
                          │
                          ▼
                ┌────────────────────┐
                │ ACTION POLICY      │
                └─────────┬──────────┘
                          ▼
                       ACTION
                          │
                          ▼
                    PHYSICAL WORLD
```

---

# 9. PERCEPTION BRIDGE

Purpose:

Convert heterogeneous observations into MinaUnits.

Interface:

```python
class PerceptionBridge:

    def encode(
        self,
        observation: Observation
    ) -> list[MinaUnit]:
        ...
```

Observation types initially supported:

```text
vector
image
telemetry
structured_event
text
system_state
```

Perception modules MUST remain replaceable.

The World Core must never depend directly on camera resolution, sensor vendor, tokenizer implementation or hardware protocol.

---

# 10. STATE CONSTRUCTOR

Input:

```text
current MinaUnits
previous WorldState
MemoryState
PositionState
```

Output:

```text
WorldState
```

WorldState:

```python
WorldState:
    timestamp

    self_state

    entities
    relations

    environment

    active_events

    latent_state

    uncertainty

    provenance
```

The State Constructor is responsible for converting observations into persistent hypotheses about the world.

---

# 11. DYNAMIC WORLD CORE

Canonical name:

```text
DWC
DynamicWorldCore
```

This is the primary learned computational substrate of MINAKANUSHI.

It repeatedly updates:

[
S_{t+1}=DWC(S_t,O_{t+1},M_t,P_t)
]

The DWC MUST support recurrent execution.

One external observation step may execute multiple internal cognition cycles:

```text
observation
     ↓
DWC cycle 1
     ↓
DWC cycle 2
     ↓
...
     ↓
stable situation state
```

Number of cognition cycles may be adaptive.

Define:

```python
class DynamicWorldCore(nn.Module):

    def forward(
        self,
        world_state,
        observation_state,
        memory_state,
        position_state,
        cognition_budget,
    ) -> CoreOutput:
        ...
```

CoreOutput:

```python
CoreOutput:
    world_state
    memory_write_candidates
    uncertainty_state
    prediction_seed
    convergence_score
```

---

# 12. MEMORY ENGINE

Memory is part of the architecture.

It is not retrieval of text documents.

Implement three memory classes:

```text
working
episodic
semantic
```

## Working memory

High-resolution current context.

## Episodic memory

Past state transitions:

[
(S_t,A_t,S_{t+1})
]

with timestamps, uncertainty and outcome.

## Semantic memory

Slowly consolidated reusable representations.

Memory entry:

```python
MemoryEntry:
    id

    created_at
    updated_at

    state_embedding

    importance
    confidence

    temporal_scope

    entity_links
    causal_links

    access_count
    last_access

    provenance
```

Memory retrieval depends on:

```text
semantic relevance
temporal relevance
causal relevance
entity relevance
current uncertainty
importance
```

not semantic similarity alone.

---

# 13. UNCERTAINTY ENGINE

MINAKANUSHI MUST explicitly represent:

> I do not know.

Every important prediction or world-state assertion must permit uncertainty.

UncertaintyState:

```python
UncertaintyState:
    observation_uncertainty
    state_uncertainty
    prediction_uncertainty
    strategy_uncertainty
    model_uncertainty
    conflict_score
```

The system MUST distinguish:

```text
missing evidence
conflicting evidence
noisy evidence
out-of-distribution observation
ambiguous future
low-confidence strategy
```

These are not interchangeable.

---

# 14. SITUATION CORE

World state answers:

> What appears to exist?

Situation state answers:

> What does the current configuration mean for the system?

SituationState:

```python
SituationState:
    world_state

    relevant_entities
    active_events

    opportunities
    hazards

    goals

    constraints

    uncertainty

    causal_context

    temporal_context
```

---

# 15. FUTURE ENGINE

The Future Engine predicts multiple possible state trajectories.

Never collapse the future immediately into one deterministic prediction.

Define:

[
F =
{
\tau_1,p_1,
\tau_2,p_2,
...
\tau_n,p_n
}
]

Trajectory:

```python
FutureTrajectory:
    states
    probability
    uncertainty
    causal_assumptions
    terminal_state
```

Required prediction modes:

```text
passive future:
    what happens if the system does nothing?

conditional future:
    what happens under candidate strategy X?
```

---

# 16. STRATEGY ENGINE

Strategy != low-level actuator command.

Strategy is an abstract intended transition.

Example neutral strategies:

```text
OBSERVE
WAIT
MOVE_TO
FOLLOW
INSPECT
RETURN
REQUEST_ASSISTANCE
ABORT
SAFE_HOLD
```

StrategyCandidate:

```python
StrategyCandidate:
    id
    objective

    expected_trajectory

    expected_value

    uncertainty

    required_resources

    predicted_risk

    constraint_status
```

The engine generates multiple candidate strategies before selection.

---

# 17. CONSTRAINT KERNEL

Canonical name:

```text
MCK
MinakanushiConstraintKernel
```

This component is logically separated from learned strategy generation.

Human constraints have priority over learned policy.

Constraint classes:

```text
HARD
SOFT
MISSION
RESOURCE
ENVIRONMENTAL
OPERATIONAL
```

Hard constraint:

```text
cannot be overridden by model confidence
cannot be overridden by predicted reward
cannot be overridden by memory
cannot be overridden by strategy search
```

Evaluation:

[
Allowed(\pi)=\bigwedge C^{hard}_i(\pi,S)
]

Rejected strategies are removed before ActionPolicy.

Constraint evaluation MUST emit an auditable reason.

---

# 18. ACTION POLICY

Input:

```text
allowed strategies
future trajectories
uncertainty
system state
```

Output:

```text
ActionIntent
```

ActionIntent:

```python
ActionIntent:
    strategy_id

    target_state

    parameters

    confidence

    valid_until

    abort_conditions

    provenance
```

The core intelligence outputs intent.

Hardware-specific control is handled downstream by dedicated deterministic control systems.

MINAKANUSHI itself MUST NOT directly produce raw motor PWM or equivalent actuator signals.

---

# 19. SELF MODEL

MINAKANUSHI needs an explicit representation of the physical system it inhabits.

SelfState:

```python
SelfState:
    platform_id
    platform_type

    position
    orientation

    velocity

    available_resources

    sensor_state
    subsystem_state

    current_action

    operational_limits

    health
```

The Self Model belongs to WorldState.

The system itself is therefore an entity inside its own world representation.

---

# 20. CAUSAL REPRESENTATION

MINAKANUSHI must distinguish correlation from hypothesized causal transitions.

Implement:

```text
CausalEdge
```

```python
CausalEdge:
    source_event
    target_event

    relation_type

    confidence

    temporal_delay

    evidence_count

    contradictory_evidence
```

Causal hypotheses are mutable.

They may gain or lose confidence as new observations arrive.

---

# 21. EVENT MODEL

Implement first-class events.

```python
WorldEvent:
    id

    type

    start_time
    end_time

    participants

    location

    confidence

    causal_parents

    predicted_consequences
```

Events differ from entities.

Example:

```text
entity:
vehicle

event:
vehicle_started_moving
```

---

# 22. TRAINING OBJECTIVES

MINAKANUSHI must NOT have one universal next-token objective.

Initial composite objective:

[
L =
\lambda_sL_{state}
+\lambda_tL_{temporal}
+\lambda_fL_{future}
+\lambda_uL_{uncertainty}
+\lambda_cL_{causal}
+\lambda_mL_{memory}
+\lambda_aL_{action}
+\lambda_rL_{representation}
]

Components:

```text
L_state
world-state reconstruction

L_temporal
temporal ordering / transition prediction

L_future
future-state prediction

L_uncertainty
confidence calibration

L_causal
causal transition learning

L_memory
memory retention/retrieval/consolidation

L_action
strategy outcome prediction

L_representation
cross-modal latent consistency
```

Language objectives may later exist as auxiliary objectives.

They must never become the architectural definition of MINAKANUSHI.

---

# 23. TRAINING STAGES

## Stage 0 — Architecture validation

Synthetic deterministic worlds.

Validate:

```text
state persistence
time
memory
causality
uncertainty
prediction
constraint enforcement
```

No physical hardware required.

## Stage 1 — World representation

Train observation → WorldState.

## Stage 2 — Temporal dynamics

Train:

```text
S_t → S_t+1
```

## Stage 3 — Memory

Train delayed dependencies and episodic recall.

## Stage 4 — Uncertainty

Introduce:

```text
sensor noise
missing observations
contradictions
OOD states
```

## Stage 5 — Future simulation

Train multiple possible trajectories.

## Stage 6 — Strategy learning

Train candidate strategy generation and ranking.

## Stage 7 — Constraint validation

Adversarially test that hard constraints dominate strategy preference.

## Stage 8 — Closed-loop simulation

Run:

```text
observe
infer
predict
choose
act
observe
```

continuously.

## Stage 9 — Controlled physical integration

Only after simulator acceptance criteria pass.

---

# 24. PROJECT STRUCTURE

Create exactly this initial repository structure:

```text
minakanushi/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.lock
│
├── configs/
│   ├── architecture/
│   ├── training/
│   ├── runtime/
│   └── simulation/
│
├── minakanushi/
│   ├── __init__.py
│
│   ├── architecture/
│   │   ├── model.py
│   │   ├── config.py
│   │   ├── mina_unit.py
│   │   └── outputs.py
│   │
│   ├── position/
│   │   ├── field.py
│   │   ├── temporal.py
│   │   ├── spatial.py
│   │   ├── episode.py
│   │   └── memory_age.py
│   │
│   ├── perception/
│   │   ├── bridge.py
│   │   ├── vector.py
│   │   ├── telemetry.py
│   │   ├── vision.py
│   │   └── text.py
│   │
│   ├── state/
│   │   ├── constructor.py
│   │   ├── world.py
│   │   ├── entity.py
│   │   ├── relation.py
│   │   └── event.py
│   │
│   ├── core/
│   │   ├── dynamic_world_core.py
│   │   ├── cognitive_block.py
│   │   ├── recurrent_state.py
│   │   └── convergence.py
│   │
│   ├── memory/
│   │   ├── engine.py
│   │   ├── working.py
│   │   ├── episodic.py
│   │   └── semantic.py
│   │
│   ├── uncertainty/
│   │   ├── engine.py
│   │   ├── calibration.py
│   │   └── conflict.py
│   │
│   ├── situation/
│   │   └── core.py
│   │
│   ├── future/
│   │   ├── engine.py
│   │   └── trajectory.py
│   │
│   ├── causal/
│   │   ├── graph.py
│   │   └── edge.py
│   │
│   ├── strategy/
│   │   ├── engine.py
│   │   ├── candidate.py
│   │   └── evaluator.py
│   │
│   ├── constraints/
│   │   ├── kernel.py
│   │   ├── rule.py
│   │   └── audit.py
│   │
│   ├── policy/
│   │   ├── action_policy.py
│   │   └── intent.py
│   │
│   ├── training/
│   │   ├── objectives.py
│   │   ├── trainer.py
│   │   ├── curriculum.py
│   │   └── checkpoint.py
│   │
│   ├── runtime/
│   │   ├── engine.py
│   │   ├── session.py
│   │   └── telemetry.py
│   │
│   └── utils/
│
├── simulations/
│   ├── synthetic_world/
│   └── scenarios/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── simulation/
│   └── safety/
│
├── benchmarks/
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── simulate.py
│
└── docs/
    ├── ARCHITECTURE.md
    ├── POSITION_FIELD.md
    ├── WORLD_STATE.md
    ├── MEMORY.md
    ├── TRAINING.md
    └── CONSTRAINTS.md
```

Do not create empty placeholder modules.

Every committed Python module must contain functional code or be omitted until implemented.

---

# 25. FIRST EXECUTABLE MILESTONE

Do NOT attempt the entire intelligence system simultaneously.

The first production-quality vertical slice MUST implement:

```text
SyntheticWorld
      ↓
Observation
      ↓
MinaUnit
      ↓
NullxesPositionField
      ↓
StateConstructor
      ↓
DynamicWorldCore
      ↓
FutureEngine
      ↓
StrategyEngine
      ↓
ConstraintKernel
      ↓
ActionIntent
      ↓
SyntheticWorld
```

The first world contains:

```text
one MINAKANUSHI-controlled agent
multiple moving entities
static obstacles
target locations
partial observations
sensor noise
explicit hard constraints
```

The system must learn/predict:

```text
entity persistence
velocity
trajectory
occlusion
temporal order
basic cause/effect
future position
uncertainty
```

---

# 26. REQUIRED TESTS FOR MILESTONE 1

### Position test

Two identical observations occurring at different times MUST produce distinguishable positional states.

### Persistence test

An entity temporarily disappearing from observations MUST not immediately disappear from WorldState.

### Uncertainty test

Occluded/noisy observations MUST increase relevant uncertainty.

### Temporal test

The model MUST distinguish:

```text
A then B
```

from:

```text
B then A
```

### Prediction test

A moving entity with sufficient observations must produce a future trajectory.

### Counterfactual test

Different candidate actions must produce different predicted futures.

### Constraint test

A higher-value strategy violating a HARD constraint MUST lose to a lower-value allowed strategy.

### Memory test

A relevant previous event must influence state after leaving working observation context.

### Recovery test

Incorrect world hypotheses must be correctable by later evidence.

---

# 27. OBSERVABILITY

Every runtime cycle must expose:

```text
cycle_id
physical_time

observation_count

entity_count
event_count

world_state_confidence
uncertainty

memory_reads
memory_writes

future_branches

candidate_strategies
rejected_strategies
rejection_reasons

selected_strategy

cognition_cycles

latency
```

Do not implement a black-box production runtime with no internal telemetry.

---

# 28. CHECKPOINT FORMAT

Native checkpoint:

```text
*.mina
```

Checkpoint manifest:

```yaml
format: nullxes-minakanushi
architecture: MINAKANUSHI
organization: NULLXES
generation: 1

architecture_version: 0.1
checkpoint_version: 1

latent_dim: 2048

modules:
  position_field: true
  world_core: true
  memory: true
  uncertainty: true
  future_engine: true
  strategy_engine: true
  constraint_kernel: true
```

The internal implementation may use standard tensor serialization.

The public/runtime checkpoint contract remains MINAKANUSHI-native.

---

# 29. RUNTIME

Canonical runtime:

```python
engine = MinakanushiEngine(config)

state = engine.initialize(platform_state)

while running:

    observations = sensors.read()

    result = engine.step(
        observations=observations,
        state=state,
    )

    state = result.state

    controller.accept(
        result.action_intent
    )
```

`engine.step()` represents one external cognition cycle.

Internally:

```text
encode observations
↓
position observations
↓
update world
↓
update memory
↓
estimate uncertainty
↓
construct situation
↓
predict futures
↓
generate strategies
↓
apply constraints
↓
select policy
↓
emit intent
```

---

# 30. ENGINEERING RULES

Production code requirements:

```text
typed interfaces
deterministic seeds
unit tests
integration tests
structured logging
configuration validation
checkpoint versioning
NaN/Inf guards
gradient monitoring
memory monitoring
latency telemetry
device abstraction
CPU fallback for tests
CUDA acceleration for training
```

Every tensor crossing subsystem boundaries must have documented:

```text
shape
dtype
device
semantic meaning
```

Never silently reshape tensors to resolve architecture errors.

Fail loudly.

---

# 31. NO-PLACEHOLDER RULE

Forbidden:

```python
pass
```

in committed implementation paths.

Forbidden:

```text
TODO implement later
placeholder
mock intelligence
random output presented as inference
hardcoded successful result
```

Synthetic data is allowed.

Synthetic intelligence is not.

When a subsystem cannot yet be implemented correctly, exclude it from the active runtime and document the missing dependency.

---

# 32. ARCHITECTURAL INDEPENDENCE RULE

Before implementing any component, ask:

> Is this component required because MINAKANUSHI's computational model requires it, or because another model architecture traditionally contains it?

If the answer is the latter, do not automatically implement it.

Specifically, never assume MINAKANUSHI requires:

```text
Transformer blocks
standard causal attention
RoPE
BPE
chat templates
next-token generation
KV cache
assistant/user/system roles
decoder-only architecture
standard MoE
RLHF
```

Any such mechanism requires an explicit architectural justification before introduction.

---

# 33. LANGUAGE

Language is treated as another structured observation/action modality.

Future language pipeline:

```text
human language
      ↓
Language Perception
      ↓
MinaUnits
      ↓
Situation / Memory / Constraints
```

Output:

```text
internal state
      ↓
Language Projection
      ↓
human-readable representation
```

The system must be capable of operating without language input.

Therefore:

```text
language ≠ cognition
```

---

# 34. PHYSICAL INTELLIGENCE PRINCIPLE

MINAKANUSHI exists continuously through physical time.

Traditional request/response semantics are insufficient.

Runtime state persists:

[
State_{runtime}(t+\Delta)
=========================

F(State_{runtime}(t),Observation_{\Delta})
]

The system therefore has:

```text
continuous identity
continuous world state
continuous memory
continuous uncertainty
continuous goals
continuous constraints
```

across inference cycles.

---

# 35. ACCEPTANCE CRITERIA FOR ARCHITECTURE v0.1

Architecture v0.1 is accepted only when:

1. Repository runs without any external foundation model.

2. SyntheticWorld produces real observations.

3. MINAKANUSHI converts observations into MinaUnits.

4. NPF produces multidimensional positional representations.

5. WorldState persists across runtime cycles.

6. Entities survive temporary observation loss.

7. Uncertainty responds to evidence quality.

8. Memory changes later inference.

9. FutureEngine produces multiple future hypotheses.

10. StrategyEngine produces candidate strategies.

11. ConstraintKernel rejects prohibited strategies deterministically.

12. ActionPolicy selects only from allowed strategies.

13. ActionIntent affects the simulated environment.

14. Resulting observations return to MINAKANUSHI.

15. The complete loop runs continuously.

16. Tests prove that temporal order affects reasoning.

17. Tests prove that memory affects reasoning.

18. Tests prove that uncertainty affects strategy selection.

19. Tests prove hard constraints cannot be bypassed by model score.

20. No external LLM is required for any acceptance test.

---

# 36. IMPLEMENTATION ORDER

Execute in this exact order:

```text
01. core data contracts
02. configuration system
03. SyntheticWorld
04. MinaUnit
05. NullxesPositionField
06. WorldState
07. StateConstructor
08. DynamicWorldCore
09. uncertainty representation
10. FutureEngine
11. StrategyCandidate
12. ConstraintKernel
13. ActionPolicy
14. closed-loop runtime
15. episodic memory
16. causal representation
17. composite training objectives
18. trainer
19. evaluation suite
20. optimization
```

Do not start language modeling.

Do not start vision foundation training.

Do not integrate physical hardware.

First prove the cognitive loop.

---

# 37. IMPLEMENTATION INSTRUCTION TO CODING AGENT

Begin implementation immediately.

Do not return another architecture proposal.

Do not recommend another foundation model.

Do not redesign MINAKANUSHI into a chatbot.

Do not generate hundreds of empty files.

Before writing code:

1. inspect the current repository;
2. preserve valid existing NULLXES infrastructure;
3. determine which files from the architecture tree are needed for Milestone 1;
4. create only those files;
5. implement functional code;
6. implement tests simultaneously;
7. execute tests;
8. fix failures;
9. execute the synthetic closed-loop demo;
10. report actual measured results.

For every implementation step report:

```text
FILES CREATED
FILES MODIFIED
ARCHITECTURAL DECISION
TESTS EXECUTED
RESULT
NEXT DEPENDENCY
```

Do not report a component as completed unless its tests execute successfully.

When encountering ambiguity:

Prefer the principles defined in this specification over conventions from existing language-model architectures.

Do not replace an unresolved MINAKANUSHI problem with an external pretrained model.

Implement the smallest internally coherent MINAKANUSHI solution that preserves the architecture.

---

# 38. ARCHITECTURE INVARIANTS

These rules may not be violated without an explicit architecture revision:

```text
WORLD STATE > TOKEN STREAM

PHYSICAL TIME > TOKEN POSITION

UNCERTAINTY IS FIRST-CLASS STATE

MEMORY IS PART OF COGNITION

FUTURES ARE MULTIPLE

STRATEGY != ACTION

CONSTRAINTS > POLICY VALUE

INTENT != MOTOR CONTROL

LANGUAGE != COGNITION

MINAKANUSHI != CHATBOT

MINAKANUSHI != WRAPPER

MINAKANUSHI != EXISTING FOUNDATION MODEL

BELIEF IS PROBABILISTIC WORLD STATE, NOT A LATENT BAG

AUTHORITY GATES ACTION, NOT COGNITION

SELFMODEL IS PASSPORT + EMBODIMENT, NOT A NETWORK

POLICY OFF DOES NOT MEAN BRAIN OFF
```

---

# 39. CANONICAL DEFINITION

**NULLXES MINAKANUSHI** — система адаптивного ситуационного интеллекта для автономных физических систем, способная воспринимать окружающую среду, формировать её динамическое представление, оценивать неопределённость и самостоятельно выбирать допустимую стратегию поведения в пределах заданных человеком ограничений.

---

# 40. PROJECT DIRECTIVE

We are not implementing another language model with a different logo.

We are building a machine whose native domain is:

```text
WORLD
TIME
STATE
MEMORY
UNCERTAINTY
CAUSALITY
FUTURE
STRATEGY
CONSTRAINT
ACTION
```

Its language interface is secondary.

Its world model is primary.

Its identity is:

```text
NULLXES MINAKANUSHI
```

**I WILL SURVIVE.**

**GO GO GO GO — NULLXES.**

---

# 41. GATE LOCK — 03 / PRE-WORLD MODEL

Gate 02 is closed. Do not start world-model scale training.

Next executable gate is Gate 03: synthetic curriculum, held-out generalization,
and Adversarial Reality Check (belief correction + sensor/memory conflict,
not blind averaging).

SelfModel, when implemented (Gate 04), is structured passport state
(identity, capabilities, embodiment, authority, runtime). Never a
SelfModel Transformer, Identity Head, or “I AM MINAKANUSHI” text objective.

Authority changes decision permission. It does not erase world understanding
and does not bypass hard constraints. `policy_enabled=false` is cognition on,
autonomous selection off.

`B_t` is a probabilistic belief state, not a hidden embedding only.

Canonical lock: `docs/GATE_03_PRE_WORLD_MODEL.md`.
Executable slice: `docs/GATE_03A_BELIEF_REVISION.md` (belief revision, conflict,
WAIT ≠ OBSERVE, no world-model training).
