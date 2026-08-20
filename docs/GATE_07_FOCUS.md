# Gate 07 — Focus Engine / Internal Attention

**Status:** implemented on `cpu_dev`
**Does not:** curiosity-as-desire, RL, emotions, drives, Gate 08 world model

Focus answers:

```text
Where is additional observation most valuable right now?
```

Not:

```text
MINA wants to look over there
```

```text
Belief
  ├── uncertainty
  ├── prediction error
  ├── novelty
  ├── unfinished / conflict (unobserved only)
  ▼
Focus Engine   (rule-based score)
  ▼
Attention Target
  ▼
Situation Core → Future → Strategy → Constraint → Action
```

Focus does **not** write `ActionIntent`. Changing FocusState must not invent
strategy IDs. Below threshold the type is `MAINTENANCE` / target `0` (NONE).

## FocusState

```text
target_id
focus_type: UNCERTAINTY_REDUCTION | PREDICTION_ERROR | NOVELTY
            | MEMORY_CONFLICT | MAINTENANCE
priority
confidence
created_at
expires_at
```

## Score

```text
focus_score = uncertainty_gain + prediction_error + novelty
            + mission_relevance - risk - resource_cost
```

No RL. No extra network. Live evidence suppresses `MEMORY_CONFLICT`.

## Forbidden in this gate

Reward hacking, RL agent, dopamine, emotions, wishes, self-preservation.
