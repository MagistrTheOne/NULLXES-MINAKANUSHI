# NULLXES MINAKANUSHI — Partner Laboratory Evaluation License

Copyright (c) 2026 NULLXES

This addendum applies to the **YMBOT-C** laboratory kit and, when NULLXES
so designates, to the partner wheel and published `*.mina` research
checkpoints used with it.

## Who may use it

Named Chinese laboratory partners of NULLXES (YMBOT-C / Yunmu–Warmcore
evaluation seats) may use MINAKANUSHI **for laboratory tests only**.

This is a grant of evaluation use. It is not a transfer of architecture
identity, not a sublicense to the public, and not permission to productize
MINA as another model family.

## You may

- run L0 / L1 from this folder;
- run L2 infer on an 80 GB-class GPU using a checkpoint NULLXES pointed you to;
- copy this kit internally inside the named partner lab;
- publish measured numbers back to NULLXES;
- cite the system as **NULLXES MINAKANUSHI** (short name MINA).

## You may not

- rebrand the architecture (no “YMBOT-C 6.8B”, no new foundation-model name);
- wrap the loop as `AutoModelForCausalLM`, a chatbot, a VLA, or an LLM API;
- emit or train PWM / raw actuator commands from this kit;
- train `minakanushi_6_8b` without written NULLXES approval;
- construct 6.8B on CPU, RTX 4090, RTX 2080, RTX PRO 6000, or 1× H100 80 GB;
- republish weights as another organization’s checkpoint;
- replace Dynamic World Core, Constraint Kernel, or identity metadata;
- present train-loss decrease as an intelligence pass.

Official capability status of published 6.8B weights remains:

```text
Research checkpoint
Accepted: NO
```

unless NULLXES issues a later written acceptance.

## Identity

Runtime identity stays in metadata, not in a prompt:

```text
architecture: MINAKANUSHI
organization: NULLXES
system_class: adaptive_situational_intelligence
native_runtime: nullxes
```

YMBOT-C is the **envelope name of this kit**. It is not the architecture.

## Warranty

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND.
