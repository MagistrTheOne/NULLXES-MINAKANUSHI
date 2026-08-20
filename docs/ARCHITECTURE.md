# MINAKANUSHI architecture (implemented)

NULLXES MINAKANUSHI is a persistent world-state machine. Language is an
optional modality, not the computational primitive.

## Loop

```text
PHYSICAL WORLD
      ↓
PERCEPTION BRIDGE
      ↓
MINA UNITS
      ↓
NULLXES POSITION FIELD
      ↓
STATE CONSTRUCTOR
      ↓
DYNAMIC WORLD CORE
      ↓
MEMORY + UNCERTAINTY
      ↓
SITUATION CORE
      ↓
FUTURE ENGINE
      ↓
STRATEGY ENGINE
      ↓
CONSTRAINT KERNEL
      ↓
ACTION POLICY
      ↓
ACTION INTENT
      ↓
PLATFORM CONTROLLER (outside MINAKANUSHI)
      ↓
PHYSICAL WORLD
```

## Learned vs not learned

Learned: perception MLPs, NPF encoders and mixer, DWC cognitive blocks,
velocity/residual heads, uncertainty projection, future residual.

Not learned: slot persistence rules, hard constraint kernel, policy fail-closed
to SAFE_HOLD when the allowed set is empty, ActionIntent contract.

## Cognitive block justification

World hypotheses are persistent entity slots. A slot queries evidence and other
slots, then updates through a learned gate. This is not decoder-only causal
token attention and is not RoPE.

## Identity

Hierarchy (do not invert):

```text
NULLXES
  → MINAKANUSHI   architecture
    → MINA        short name
      → instance  runtime / checkpoint
```

Checkpoint and config identity is `architecture=MINAKANUSHI` (short name
`MINA`), `architecture_id=nullxes.minakanushi`, `organization=NULLXES`,
`native_runtime=nullxes`. No chat template. No “I AM MINAKANUSHI” language
objective. SelfModel is a structured passport; PersonaModel is presentation
only. Persona / voice / face sit above cognition and do not drive belief.
See `docs/GATE_04_IDENTITY.md`.

## Belief

World update is `B_t = F(B_{t-1}, O_t, M_t, P_t, U_t)`. `B_t` is a
probabilistic belief: position/velocity **mean + std**, existence
probability, and prediction confidence. `latent_state` is the learned
carrier, not the belief object. Occupied is slot allocation; existence is
the soft “I think this is real.” See `docs/GATE_05_BELIEF.md`.

## Authority

Authority changes decision permission. It does not erase world understanding
and does not bypass the constraint kernel. `policy_enabled=false` keeps
perception, memory, futures, and risk estimation on; autonomous ActionPolicy
select goes off (fail-closed SAFE_HOLD).

## Gates

Organism order (Maga). Do not implement a later gate in an earlier PR.

| Gate | Work | Status |
|---:|---|---|
| 02 | train loop | done |
| 03A | reality correction | done |
| 04 | existence (Self / Authority) | done |
| 05 | Belief Engine | done |
| 06 | Memory as Experience | done |
| 07 | Focus / Attention Selection | done |
| 08 | Active World Model (`Belief_t` + Action → `Belief_{t+1}`) | done |
| 08.5 | Dataset Reality Check (replay / inspector / balance / causal) | done |
| 09 | Autonomous Runtime | done |
| 10 | Embodiment adapters | not this PR |
| 11 | Language interface | not this PR |

Do not instantiate `research_v01`. Do not add vision, LLM, or extra deps.
