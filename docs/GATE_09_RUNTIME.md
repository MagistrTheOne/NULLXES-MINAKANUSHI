# Gate 09 — Autonomous Runtime

**Status:** accepted on `cpu_dev`. Foundation freeze candidate.
**Does not:** desire, emotion, motivation, “MINA lives as a person.”
**Tag:** `MINAKANUSHI-v0.1-foundation` after this gate. RunPod is next, not this file.

Gate 09 is engineering continuity:

> the system stops being only `engine.step(observation)` and becomes a
> process with cycle state that survives time, shutdown, and restore.

```text
SelfModel     → who I am
WorldState    → what is around
RuntimeState  → where this process is now
```

## Cycle

`MinakanushiRuntime.cycle()` owns the loop. `MinakanushiEngine.step()` remains
one cognition tick inside it.

```text
observe
 → update WorldState / Belief
 → memory
 → focus
 → futures
 → strategies
 → constraints
 → authority
 → ActionIntent
 → platform.execute (ActionIntent, not PWM)
 → observe consequence on the next cycle
 → write experience
```

No operator command is required for cognition. Authority decides whether an
autonomous intent may execute.

## Authority modes

| Mode | Cognition | Executable intent |
|---|---|---|
| AUTONOMOUS | on | ActionPolicy over kernel-allowed set |
| ADVISORY | on | SAFE_HOLD; `strategy_proposal` in telemetry |
| DIRECTED | on | operator intent if kernel-allowed, else SAFE_HOLD |
| MANUAL | on | SAFE_HOLD |
| SAFE_HOLD | on | SAFE_HOLD |

`policy_enabled=false` does not disable belief, memory, focus, or futures.
It does not mint a kernel-rejected MOVE_TO.

## Restore

```text
cycle N → checkpoint → shutdown → restore → cycle N+1
```

Restored: SelfModel, WorldState/Belief, Memory (including write cursor),
Focus, Authority, RuntimeState, last prediction, last intent, SyntheticWorld
plant snapshot.

## Metrics

`runtime_cycles`, `belief_updates`, `memory_writes`, `focus_changes`,
`prediction_updates`, `action_attempts`, `authority_blocks`,
`checkpoint_restores`, `experience_records`.

## Not this gate

Git tag `MINAKANUSHI-v0.1-foundation` only after this gate is accepted.
Then: push, clean clone, RunPod, `gpu_train_v01`. Do not instantiate
`research_v01` here. Gate 10 is embodiment adapters.
