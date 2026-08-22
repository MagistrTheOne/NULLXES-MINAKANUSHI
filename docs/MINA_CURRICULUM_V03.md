# MINAKANUSHI Status Core Curriculum v0.3

Data only. Architecture freeze `7aba976` stays. Do not touch DWC / dims / losses.

```text
dataset/mina_6_8b_v03
physics:     32 frames
agency:      32 frames
causality:   64 frames
embodiment:  64 frames
```

Why: v0.2 length-11 was see → think → answer. v0.3 is observe → error → evidence → revision → new future.

Corrections are typed (`wrong_velocity`, `hidden_object`, `agent_changes_goal`, `unexpected_physics`, `sensor_delay`, `wrong_intent`), not more random noise. Length > 12 fires a second real cause (late turn / accel flip / re-hide evidence), so 1000 episodes can reach 2000–5000 corrections.

Each episode stores WAIT / MOVE_TO / FOLLOW / AVOID forks from the same world seed (`counterfactuals.future_diversity > 0`). Trainer still trains WAIT vs MOVE_TO; the JSON is the food for later.

```text
python scripts/generate_6_8b_curriculum.py --root dataset/mina_6_8b_v03 --n 250
python scripts/split_heldout.py --root dataset/mina_6_8b_v03
python scripts/audit_curriculum.py --root dataset/mina_6_8b_v03 --gate
```

Held-out is `episode_index % 10 == 9` by `(seed, scenario, episode_index)`, not a file shuffle: 900 train / 100 held-out. JSON stays in phase folders; only `train/index.jsonl` and `heldout/index.jsonl` are added.

Extended audit (n≥1000) also fails if a revision type is zero, if one action ≥95%, or if `future_diversity` has no spread (min/max/mean/std).

Gate before H200:

```text
1000+ episodes
32/64 frame trajectories
correction_count >= 2000
future_diversity > 0
pwm=false
```

Resume after that:

```text
--config configs/training/mina_6_8b_v03.yaml
--resume minakanushi_stage0_step128.mina
```

Phase 1 is 1000 steps then STOP (`docs/MINA_TRAINING_CONTRACT_v03.md`). Train reads `dataset_split: train`.

Safetensors remain a Hub mirror: `scripts/export_safetensors.py`. Canonical is `*.mina`.

Train mix is not the on-disk 250/250/250/250 audit split. `sampler_mode: auto` does warm then intelligence. Hidden-correction L2/L3 are in the generator (`docs/MINA_OPTIMIZATION_V031.md`).
