# MINA 6.8B — model folder

```text
NULLXES MINAKANUSHI 6.8B
short name: MINA
HF: MagistrTheOne/MINAKANUSHI-6.8B
```

**Status:** Status Core (Researched). Hugging Face card: `HF_README.md`.

Weights are not stored in git. The trained `*.mina` lives on Hugging Face:

https://huggingface.co/MagistrTheOne/MINAKANUSHI-6.8B

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
| Weights | Hugging Face `checkpoints/minakanushi_stage0_step64.mina` |
| Spec | `docs/MINA_6_8B_TRAINING.md` |
| Card | `models/MINA-6.8B/HF_README.md` |

Do not `MinakanushiSystem(this)` on a laptop or the Stage A 6000 pod.

Pilot package for Yunmu: runtime + this config + safety/authority docs +
SyntheticWorld episodes. Warmcore finetune stays on NULLXES episode format.
The architecture identity in the `*.mina` manifest does not change.
