# WorldState

WorldState is a probabilistic belief about reality, not a copy of the current
observation and not a hidden embedding only.

Native target:

```text
B_t = F(B_{t-1}, O_t, M_t, P_t, U_t)
```

Belief is mean + distribution + existence. `latent_state` is the learned
carrier, not the public belief. Occupied is slot allocation; existence is
the soft probability that the hypothesized entity is real. A world model
that predicts only `future_xy` or the next image is the wrong object.

Tensors (batch B, slots N, dim D, uncertainty channels U):

| Field | Shape | Meaning |
|---|---|---|
| latent_state | [B, N, D] | learned carrier (not the belief) |
| entity_xy | [B, N, 2] | position mean |
| xy_std | [B, N, 2] | position std (clamped min) |
| entity_vel | [B, N, 2] | velocity mean |
| vel_std | [B, N, 2] | velocity std |
| existence | [B, N] | P(entity is real), in (0, 1] while occupied |
| pred_confidence | [B, N] | confidence in the dynamics prior |
| occupied | [B, N] | slot allocated |
| entity_id | [B, N] | identity, 0 empty |
| uncertainty | [B, N, U] | typed uncertainty channels |
| age_unobserved | [B, N] | steps without evidence |

`WorldState.as_belief()` is the public view: position/velocity mean+std,
existence, last observation age, prediction confidence. Cause history is
the `corrections` tuple (Gate 03A), not retrieval.

Lifecycle: create on first evidence, update on match by entity_id, persist
while `age_unobserved <= persistence.steps`, retire after that, correct when
later evidence for the same id arrives.

Self is slot 0 (entity_id=1, kind=agent).

Belief revision (Gate 03A): returning evidence after a gap writes a
`CorrectionEvent` and must not be a 50/50 average of stale memory and the
new measurement. Unobserved occupied slots coast on velocity while
uncertainty grows, `xy_std` inflates, and existence decays toward retirement.
See `docs/GATE_03A_BELIEF_REVISION.md` and `docs/GATE_05_BELIEF.md`.
