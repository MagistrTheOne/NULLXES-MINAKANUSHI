# RunPod training pipeline (preparation only)

Do **not** start this until Gate 08 contract tests pass on CPU.

## Order

1. Local CPU: `python scripts/generate_dataset.py`
2. Local CPU: `python -m pytest tests`
3. Local CPU: `python scripts/train.py` on `configs/training/stage0_validation.yaml`
4. RunPod GPU: `gpu_train_v01` mixed synthetic curriculum, checkpoint, profiling
5. Only then H100/H200 and `research_v01`

## Not now

- Instantiate `research_v01` on the CPU box
- Millions of samples
- Vision / language deps

Architecture stays MINAKANUSHI. Dataset is SyntheticWorld v1, not an
external foundation-model corpus.
