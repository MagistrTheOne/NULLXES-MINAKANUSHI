# WorldState

WorldState is a probabilistic belief about reality, not a copy of the current
observation and not a hidden embedding only.

Native target:

```text
B_t = F(B_{t-1}, O_t, M_t, P_t, U_t)
```

`latent_state` carries the learned hypothesis. Belief also includes identity,
kinematics, confidence, typed uncertainty, occupancy, and age_unobserved.
A world model that predicts only `future_xy` or the next image is the wrong
object.

Tensors (batch B, slots N, dim D, uncertainty channels U):

| Field | Shape | Meaning |
|---|---|---|
| latent_state | [B, N, D] | persistent hypotheses |
| entity_xy | [B, N, 2] | hypothesized position |
| entity_vel | [B, N, 2] | hypothesized velocity |
| occupied | [B, N] | slot in use |
| entity_id | [B, N] | identity, 0 empty |
| uncertainty | [B, N, U] | typed uncertainty |
| age_unobserved | [B, N] | steps without evidence |

Lifecycle: create on first evidence, update on match by entity_id, persist
while `age_unobserved <= persistence.steps`, retire after that, correct when
later evidence for the same id arrives.

Self is slot 0 (entity_id=1, kind=agent).
