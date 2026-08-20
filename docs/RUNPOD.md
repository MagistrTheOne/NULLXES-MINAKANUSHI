# RunPod training pipeline (preparation only)

Do **not** start this until Gate 09 autonomous runtime is accepted, then tag
`MINAKANUSHI-v0.1-foundation`.

## Order

1. Local CPU: Gate 08.5 dataset reality — done
2. Local CPU: Gate 09 autonomous runtime
3. Git tag `MINAKANUSHI-v0.1-foundation`
4. Push `main`, clean clone
5. RunPod GPU: `gpu_train_v01` mixed synthetic curriculum, checkpoint, profiling
6. Only then H100/H200 and `research_v01`

## Not now

- Instantiate `research_v01` on the CPU box
- Millions of samples
- Vision / language deps

Architecture stays MINAKANUSHI. Dataset is SyntheticWorld v1, not an
external foundation-model corpus.
