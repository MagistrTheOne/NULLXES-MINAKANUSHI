# Gate 08 — Active World Model

**Status:** implemented on `cpu_dev` (contract + kinematics). Not `gpu_train_v01`.
**Does not:** scale training, `research_v01`, H100, millions of samples.

## Contract

```text
Belief(t) + Action(t) → Belief(t+1)
```

Not `state → future_xy`. Not `observation → next observation`.

```text
              Belief(t)
                  |
        ┌─────────┴─────────┐
        ▼                   ▼
 Passive Future       Action-conditioned Future
 (WAIT / hold)         (MOVE_TO / …)
        └─────────┬─────────┘
                  ▼
          Future Belief(t+1)
                  ▼
          compare with reality
                  ▼
     ActionOutcomeRecord → Experience → Lesson
```

`FutureEngine.predict_belief` rolls **belief tensors** (xy, vel, std, existence).
Only the agent slot takes the action. Other entities coast. Current WorldState
is not mutated.

## Sub-gates

| ID | Claim |
|---|---|
| 08A | Passive dynamics: WAIT coasts movers; brake stays; existence decays |
| 08B | WAIT vs MOVE_TO → different agent future belief |
| 08C | Two MOVE_TO targets → two future beliefs |
| 08D | After act: predicted vs observed → ActionOutcomeRecord |

## Metrics

Action influence, counterfactual separation, causal consistency (action must
not invent mover motion), prediction calibration.

## Dataset / RunPod

See `docs/DATASET_V1.md` and `docs/RUNPOD.md`. Generate on CPU first.
Do not start `gpu_train_v01` until the contract tests pass.
