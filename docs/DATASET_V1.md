# NULLXES SyntheticWorld Dataset v1

CPU contract. Not a dump of millions of frames.

```text
dataset/
├── train/
├── validation/
├── composition/
├── ood/
└── counterfactual/
```

Each episode JSON:

```json
{
  "episode_id": "",
  "seed": 123,
  "self_state": {},
  "observations": [],
  "world_states": [],
  "belief_states": [],
  "actions": [],
  "future_branches": [],
  "events": [],
  "corrections": []
}
```

Replay (Gate 08.5A): same `(seed, scenario, episode_index, length)` yields
bitwise-identical canonical JSON for world, observations, actions, events,
futures, and outcomes.

Inspector (08.5B):

```text
python scripts/inspect_episode.py dataset/ood/<episode>.json
```

Balance (08.5C):

```text
python scripts/dataset_balance.py dataset
```

Generate:

```text
python scripts/generate_dataset.py --root dataset --n 4 --length 8
```

`belief_states` are teacher visibility confidences (inspectable). MINA
posterior tensors are filled by a closed-loop recorder, not by this writer.

`occlusion` events are in-range unseen bodies (hidden or line-of-sight).
Bodies beyond `sensor_range` are `out_of_range`, not occlusion.
`gone_forever` removes the entity from ground truth and emits `disappearance`.
