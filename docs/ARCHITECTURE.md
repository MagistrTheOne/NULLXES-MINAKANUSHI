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
No “I AM MINAKANUSHI” language objective. Self-knowledge is a structured
SelfModel (passport + embodiment + authority), not a network that classifies
identity. See `docs/GATE_03_PRE_WORLD_MODEL.md`.

## Belief

World update is `B_t = F(B_{t-1}, O_t, M_t, P_t, U_t)`. `B_t` is a
probabilistic belief state, not a hidden embedding only. `latent_state` is
the learned carrier; identity, kinematics, confidence, typed uncertainty,
and persistence are part of the belief.

## Authority

Authority changes decision permission. It does not erase world understanding
and does not bypass the constraint kernel. `policy_enabled=false` keeps
perception, memory, futures, and risk estimation on; autonomous ActionPolicy
select goes off (fail-closed SAFE_HOLD).

## Gates

| Gate | Status |
|---:|---|
| 02 | closed (loop, grads, checkpoint, runtime) |
| 03 | synthetic curriculum + generalization + adversarial reality — next |
| 04 | SelfModel + Authority + operator modes (structured, not a net) |
| 05 | world belief objectives |
| 06 | memory / uncertainty validation |
| 07 | `gpu_train_v01` scaling |
| 08 | H100/H200 preparation |
