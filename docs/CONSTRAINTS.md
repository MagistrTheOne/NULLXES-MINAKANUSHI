# Constraint kernel

Human hard constraints outrank learned policy value.

Mandatory order:

```text
candidate strategy
      → constraint evaluation
      → allowed / rejected + audit
      → policy selection among allowed only
```

Hard rules in Milestone 1:

- stay_in_arena
- no_enter_restricted_zone
- no_collide_obstacle
- max_speed

A higher-value MOVE_TO into a restricted zone is rejected. ActionPolicy never
sees it. If the allowed set is empty, policy fail-closes to SAFE_HOLD.

Authority may disable autonomous selection (`policy_enabled=false`). That
does not disable perception, memory, futures, or this kernel. Authority
cannot bypass HARD rules and cannot keep a rejected strategy executable.

Rejection reasons are telemetry (`rejected_strategies`, `rejection_reasons`).
