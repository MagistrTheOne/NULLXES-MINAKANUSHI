# Dataset pipeline v0.2

```text
ARCHITECTURE FREEZE — v0.2 changes loaders and data, not DWC / dims / layers.
```

SOURCE OF TRUTH:

```text
dataset/mina_6_8b/{physics,agency,causality,embodiment}/*.json
dataset/mina_6_8b/index.jsonl
```

Causal episode, not a token dump:

```text
state0 → decision → action → consequence → error → revision → new prediction
```

```text
JSON
  → JsonEpisodeDataset (stream index.jsonl)
  → record_to_episode
  → training_frame
  → Trainer.unroll
  → MINA
```

Required keys: `episode_id`, `seed`, `scenario`, `observations`, `world_states`, `belief_states`, `actions`, `events`, `corrections`, plus 6.8B extras `phase`, `curriculum`, `transitions`, `embodiment`.

HF datasets are adapters into Observation / WorldState / ActionIntent / Event. They are not the train source. See `docs/HF_WORLD_MODEL_ADAPTERS.md`.
