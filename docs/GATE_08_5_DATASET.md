# Gate 08.5 — Dataset Reality Check

**Status:** implemented on `cpu_dev`. Not Gate 09. Not `gpu_train_v01`.
**Question:** is SyntheticWorld a teacher, or just a random movie?

Gate 08 proved `Belief(t)+Action → Belief(t+1)`. This gate proves the
synthetic episodes that would train that contract are **replayable,
inspectable, balanced, and causal**.

```text
08.5A replay        same (seed, scenario, config) → same dump
08.5B inspector     human readout, not a trainer
08.5C balance       not 90% constant velocity
08.5D causal sanity WAIT vs MOVE_TO; distant entity unchanged
```

## 08.5A Replay

```text
seed + scenario + episode_index + length + config
        ↓
generation #1  ==  generation #2
```

Checked fields: `world_states`, `observations`, `actions`, `events`,
`future_branches`, `outcomes`, `corrections`. Identity is canonical JSON
(`sort_keys`, no whitespace drift).

## 08.5B Inspector

Not training. `scripts/inspect_episode.py path.json` prints:

```text
TIME 0
Entities:
Belief:
Action:
Predicted:
REALITY:
Correction:
```

`belief_states` here are **teacher visibility confidences**, not MINA
posterior tensors. Closed-loop recorder can replace them later.

## 08.5C Balance

Counts: scenario, event, occlusion, action, correction, conflict.
A split is rejected as collapsed if `const_velocity ≥ 90%` of episodes.

```text
python scripts/dataset_balance.py dataset
```

## Event taxonomy

`occlusion` = in-range and unseen (hidden or blocked).
`out_of_range` = beyond sensor range. Not a correction.
`disappearance` = entity left ground truth (`gone_forever`).

Same world seed and initial placement. Episode 1: `WAIT`. Episode 2:
`MOVE_TO`. Agent trajectories must diverge. A distant non-interacting
mover with zero velocity must stay numerically identical.

This is teacher-world causality. MINA `predict_belief` causality is Gate 08.

## Not this gate

Persistent `RuntimeState`, `cycle()`, checkpoint restore loop — Gate 09.
See `docs/GATE_09_RUNTIME.md`.

Do not tag `MINAKANUSHI-v0.1-foundation` until Gate 09 is accepted.
Do not start RunPod.
