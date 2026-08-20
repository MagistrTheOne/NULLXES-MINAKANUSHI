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

Replay: same `(seed, scenario, episode_index, length)` yields the same
`world_states`.

Generate:

```text
python scripts/generate_dataset.py --root dataset --n 4 --length 8
```

`belief_states` / `future_branches` stay empty until a closed-loop recorder
fills them. That is allowed in v1.
