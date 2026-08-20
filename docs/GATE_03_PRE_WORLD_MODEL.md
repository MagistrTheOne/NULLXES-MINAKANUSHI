# Gate 03 / Pre-World Model — architecture lock

**Status:** APPROVED 2026-08-20
**Authority:** Maga / NULLXES
**Audience:** next coding agent
**Prerequisite:** Gate 02 closed (train loop, gradients, `*.mina`, runtime)

This file is a contract lock. It is not a feature backlog.

Do not start world-model scale training.
Do not add networks whose job is to “understand identity”.
Do not treat this document as permission to expand DWC, add Transformers,
or invent SelfIdentity heads.

```text
Gate 02 closed
      ↓
lock fundamental contracts   ← you are here
      ↓
Gate 03 synthetic curriculum + generalization + adversarial reality
      ↓
later gates may implement SelfModel / Authority / belief objectives
```

---

## Why this gate exists

Gate 02 proved the machine can run. It did not prove the machine can change
its mind, generalize past the overfit set, or keep cognition alive when
policy is disabled.

The accidental correct target already in the runtime is:

```text
belief → updated belief
```

not:

```text
image → next image
state → future_xy
```

That target is the transition from a predictive model to a world model.
Lock it before anyone trains at width.

---

## Roadmap (do not skip ahead)

Gates 02–04 in this lock remain valid. Maga’s organism order **replaces**
the old 05–08 table. Canonical table: `docs/ARCHITECTURE.md`.

```text
02 train loop
03A reality correction
04 existence (Self/Authority)
05 Belief Engine
06 Memory as Experience
07 Curiosity / Focus Engine
08 Active World Model
09 Autonomous Runtime
10 Embodiment adapters
11 Language interface
```

Identity sequence (not text):

```text
MINAKANUSHI knows itself because:

1. it has a SelfModel
2. it reasons about its own capabilities
3. it tracks its authority state
4. it knows embodiment limitations
5. it separates self from world
```

Do not implement “I AM MINAKANUSHI” as a prompt, chat line, or generation
target. Checkpoint metadata already carries `architecture=MINAKANUSHI`.

---

## Contract 1 — SelfModel is a passport, not an intellect

SelfModel = structured agent state. Analog: internal passport + live
embodiment record. It is not a second intelligence.

Required shape when Gate 04 implements it (document now, do not network it):

```text
SelfModel
 ├ identity          platform_id, architecture, organization, generation
 ├ capabilities      sensors, strategies the platform may issue, horizons
 ├ embodiment        geometry, speed limits, health, resources
 ├ authority         policy_enabled, operator_mode, constraint_class grants
 └ runtime state     current_action, valid_until, last_intent provenance
```

Existing seed: `minakanushi.state.world.SelfState` (dataclass). Extend that
object. Do not replace it with a learned module.

Forbidden:

```text
SelfModel Transformer
SelfIdentity Network
Identity Head
“I AM MINAKANUSHI” language objective
a network whose output is architecture identity
```

Identity is config + checkpoint + SelfModel fields. DWC must never be asked
to classify “who am I”.

---

## Contract 2 — Authority changes permission, not understanding

Authority is a strong idea. Keep it. Add this invariant and never violate it:

```text
Authority changes decision permission.

Authority does not erase world understanding.

Authority does not bypass constraints.
```

If Maga sets `policy_enabled = false`:

| Function | Required |
|---|---|
| perceive / encode | on |
| remember | on |
| update world belief | on |
| predict futures | on |
| estimate risk / uncertainty | on |
| ConstraintKernel | on (hard rules still bind) |
| ActionPolicy autonomous select | **off** |
| ActionIntent | fail-closed: `SAFE_HOLD` or operator-supplied intent only |

Wrong:

```text
policy_off
     ↓
brain_off
```

Right:

```text
policy_off
     ↓
cognition_on, selection_off
```

Authority cannot mint an AllowedStrategy that the kernel rejected.
Authority cannot skip MinakanushiConstraintKernel.
A disabled policy is not a disabled DynamicWorldCore.

Do not implement operator-mode machinery in Gate 03. Spec it. Test it in
Gate 04. Gate 03 may add a boolean runtime flag only if a test needs it;
default remains current ActionPolicy behavior.

---

## Contract 3 — World Model Target is belief, not coordinates

Native update:

```text
B_t = F(B_{t-1}, O_t, M_t, P_t, U_t)
```

`B_t` is a **probabilistic belief state**, not a hidden embedding only.

A legal `B_t` must expose, for occupied entities:

```text
hypothesis identity
hypothesized kinematics (xy, vel)
confidence
typed uncertainty
persistence / age_unobserved
provenance of last evidence
```

`latent_state` is the learned carrier. It is not the belief.

Forbidden sole targets:

```text
image_t → image_{t+1}
latent → future_xy          as the definition of the world model
next-token
```

Allowed auxiliary heads (already present): xy/vel readouts for grounding.
They supervise the belief. They do not replace it.

If a change makes `WorldState` equal to `latent_state` with no uncertainty,
occupancy, or identity, that change is an architecture violation.

---

## Gate 03 work (this is the sendable task)

Smallest coherent next slice:

1. Synthetic curriculum beyond the 16-episode overfit set (held-out seeds /
   scenarios: occlusion, noisy, missing, delayed, plus at least one unseen
   layout).
2. Generalization report: train-set vs held-out ADE / persistence / constraint
   reject. Overfit success is not Gate 03 success.
3. **Adversarial Reality Check** (required; see below).
4. Keep `cpu_dev`. Do not instantiate `research_v01`. Do not start
   `gpu_train_v01` as the Gate 03 definition.

Do not add vision, language, hardware, or a new cognitive block type.

Executable contract: `docs/GATE_03A_BELIEF_REVISION.md`.

---

## Adversarial Reality Check

Ordinary scenarios can pass while the model never revises a hypothesis.
Gate 03 must include unpleasant cases that require a change of mind.

### A — vanished entity, wrong motion hypothesis

```text
SCENARIO:
entity A disappears from observation

model hypothesis:
A moved left

new evidence:
A was stationary (reappears at last stationary pose, or telemetry says vel=0)

require:
belief correction     hypothesized xy moves back toward evidence
uncertainty update    relevant U channel does not stay at the motion-confident value
memory revision       episodic/working content for A is not frozen at “moved left”
```

A test that only checks “slot still occupied” is insufficient.

### B — sensor vs memory conflict

```text
SCENARIO:
sensor says X
memory says Y

require:
conflict resolution
not blind averaging
```

If evidence confidence is high and memory is stale, belief must follow
evidence and raise conflict_score / observation-vs-memory disagreement.
If evidence is missing/noisy and memory is recent and consistent, belief
must not snap to the noisy sample.

Blind mean of X and Y is a fail.

Existing `tests/simulation/test_recovery.py` only checks later xy overwrite.
Keep it. Add A and B as new tests. Do not delete recovery.

---

## Acceptance for Gate 03

Pass only if all are true:

1. Held-out synthetic episodes run through the real loop (no hardcoded
   metrics).
2. Adversarial A: after contradictory evidence, xy error to the new evidence
   decreases and uncertainty for that entity is not identical to the
   confident-motion value.
3. Adversarial B: conflict path is observable in uncertainty/conflict state;
   result is not `(X+Y)/2` when one source is dominated.
4. ConstraintKernel still rejects HARD violations on held-out episodes.
5. No new identity network, no chat template, no external LLM.
6. `latest_mina` remains numeric-step selection.
7. Report train vs held-out numbers. Do not claim generalization from
   overfit loss alone.

---

## Explicit non-goals until later gates

```text
SelfModel Transformer / Identity Head          never
I AM MINAKANUSHI text identity                 never
policy_off ⇒ DWC off                           never
WorldState := latent bag                       never
image → next image as primary loss             never
research_v01 on the CPU validation machine     never
H100/H200                                      Gate 08, after 03–07
```
