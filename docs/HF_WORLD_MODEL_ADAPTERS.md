# Hugging Face world-model adapters

```text
ARCHITECTURE FREEZE
SOURCE OF TRUTH: dataset/mina_6_8b
HF: adapter only. Do not train on raw Hub dumps.
```

Map foreign rows into Observation / WorldState / ActionIntent / Event.

## Chosen

| Repo | When | Map |
|---|---|---|
| Native SyntheticWorld `dataset/mina_6_8b` | **this cycle train** | already MinaUnit-native |
| [farama-minari/D4RL](https://huggingface.co/datasets/farama-minari/D4RL) and [farama-minari/mujoco](https://huggingface.co/datasets/farama-minari/mujoco) | after JSON loader, adapter #1 | `(s,a,s')` → telemetry Observation, ActionIntent, teacher world |
| [jxu124/OpenX-Embodiment](https://huggingface.co/datasets/jxu124/OpenX-Embodiment) | Yunmu later | sensors → Observation; motors stay on their controller |

Fallback timeseries: [yixuan1999/interactive-world-sim-mujoco-data](https://huggingface.co/datasets/yixuan1999/interactive-world-sim-mujoco-data).

## Rejected this cycle

NVIDIA PhysicalAI WorldModel *Scenes (video / Cosmos). LeRobot RGB Open-X mirrors. Pixels-in / tokens-in is Gate 9+ perception, not Status Core v0.2. Cosmos remains an idea-direction for `docs/experiments/MINA_V2_MULTIMODAL.md` (state prediction / MinaUnits / ActionIntent), not a train source and not a weight.
