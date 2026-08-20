# MINA 6.8B — model folder

```text
NULLXES MINAKANUSHI 6.8B
short name: MINA
```

**Status:** PRE-TRAIN GATE. Frozen at `7aba976`. Weights not in this folder.

```text
MINA
|-- SelfModel
|-- Authority
|-- Belief
|-- Memory
|-- Focus
|-- Active World Model
|-- Runtime
|-- ActionIntent
└── Humanoid / physical systems interface   (Gate 10, adapter — not PWM)
```

| | |
|---|---|
| Config | `architecture.yaml` (copy of `configs/architecture/minakanushi_6_8b.yaml`) |
| d / DWC / slots | 4096 / 32 / 512 world + 1024 memory |
| Params (formula) | **6 799 130 646** |
| Weights | not in this folder until a `*.mina` is trained |
| Spec | `docs/MINA_6_8B_TRAINING.md` |

Do not `MinakanushiSystem(this)` on a laptop or the Stage A 6000 pod.

Pilot package for Yunmu: runtime + this config + safety/authority docs +
SyntheticWorld episodes. Trained 6.8B checkpoint is a later joint step
(optional Warmcore finetune on NULLXES datasets).
