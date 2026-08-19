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

Checkpoint and config identity is `architecture=MINAKANUSHI`,
`organization=NULLXES`, `native_runtime=nullxes`. No chat template.
