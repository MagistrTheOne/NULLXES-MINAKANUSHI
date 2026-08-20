# Gate 03A — Belief Revision + Generalization Contract

**Status:** executable slice under `docs/GATE_03_PRE_WORLD_MODEL.md`
**Does not train a world model.** Does not add identity networks.

```text
CERBER:        I saw
MINAKANUSHI:   I believe what is happening
Future:        I think what may happen
Strategy:      I choose an allowed act
Controller:    I move hardware
```

## Belief is not memory

```text
PAST      → Memory     "what was"
CURRENT   → Belief     "what is probable now"
POSSIBLE  → Future     "what may be"
```

Mixing these three is an LLM-style context window. Forbidden.

Native update remains `B_t = F(B_{t-1}, O_t, M_t, P_t, U_t)` where `B_t` is a
probabilistic belief (identity, kinematics, confidence, typed uncertainty,
persistence), not `latent_state` alone.

## Primitive

`CorrectionEvent` in `minakanushi/state/correction.py`:

```text
previous belief + new evidence + difference + correction magnitude
```

Recorded on hypothesis revision, not on first detection and not on ordinary
tracking of a currently visible entity.

## Tests this gate must pass

1. Hidden entity correction — disappear, uncertainty up / confidence down,
   then stationary evidence revises the “moving left” hypothesis and writes
   a CorrectionEvent.
2. Conflict resolution — stale memory vs fresh evidence is not `(A+B)/2`.
3. WAIT ≠ OBSERVE in FutureEngine action vectors (zero plant velocity still).
4. False persistence after gone-forever is bounded by `persistence.steps`.
5. `L_uncertainty` uses state channel 6, not the mean of all typed channels.
6. Memory effect is measured on latents of unobserved slots, not all xy.

## Eval

```text
python -m pytest tests/unit/test_belief_revision.py tests/unit/test_wait_alias.py tests/simulation/test_adversarial_reality.py tests/simulation/test_conflict_resolution.py tests/simulation/test_recovery.py
python scripts/gate03a_eval.py
```

No `research_v01`. No `gpu_train_v01`. No SelfModel network.
SelfModel + Authority stay Gate 04, after belief revision is proven.
