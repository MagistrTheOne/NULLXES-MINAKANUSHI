# MINA Optimization Pass v0.3.1

Loop quality. Not a new MINA. Architecture freeze `7aba976` stays.

Do not touch DWC, latent_dim, slots, world state, loss architecture, or layers.

```text
1. Freeze step128          scripts/freeze_step128.py
2. Resume replay           scripts/audit_resume.py + tests/unit/test_resume_replay.py
3. HF safetensors test     scripts/gate_v031_export.py
4. Curriculum sampler      sampler_mode: auto in mina_6_8b_v03.yaml
5. 100-episode validation  scripts/gate_v031_validate.py
6. H200 full training      only after 1–5
```

## Gate 0 — baseline

```text
python scripts/freeze_step128.py --mina minakanushi_stage0_step128.mina --out artifacts/v031/step128
```

Writes `sha256.txt` and `metrics_before.json` from the zip manifest. Does not construct 6.8B. `reference_inference_before.pt` is cpu_dev-only; research-scale inference is `load_mina` on H200/B300.

## Gate 1 — resume

A real continuation has AdamW + scheduler + RNG + `dataset_cursor`. Missing optimizer is a clone, and `apply_resume` refuses it.

```text
python scripts/audit_resume.py --mina minakanushi_stage0_step128.mina
```

CPU proof: `test_resume_replay.py` — two resumes of step1 produce the same step2 loss; a weights-only Adam is a different pupil.

## Gate 2 — sampler

Uniform 250/250/250/250 stays on disk for audit.

Train mix:

```text
warm (first warm_steps of this job):
  physics 40%  causality 30%  agency 20%  embodiment 10%

intelligence (after):
  causality 40%  agency 30%  embodiment 20%  physics 10%
```

v0.3 resume YAML: `sampler_mode: auto`, `warm_steps: 16`.

## Gate 3 — hidden correction levels

```text
L1  object disappeared          hidden_correction
L2  object changed velocity     hidden_correction_l2
L3  object changed intent       hidden_correction_l3
```

Length <= 12 stays L1 so Gate 03 does not move.

Regenerate v0.3 if you want L2/L3 in the JSON pack:

```text
python scripts/generate_6_8b_curriculum.py --root dataset/mina_6_8b_v03 --n 250
python scripts/audit_curriculum.py --root dataset/mina_6_8b_v03 --gate
```

## Gate 4 — CPU loss probe

```text
python scripts/gate_v031_loss_probe.py --steps 32
```

cpu_dev. Watch `future` / `revision` / `memory`. New metrics, not new losses:

```text
memory_ade_on
memory_ade_off
memory_helps_future     1 if ADE(on) < ADE(off)
```

`memory_effect_delta` is still latent change. ADE answers whether memory improves the future.

## Gate 5 — export / registry

```text
python scripts/gate_v031_export.py --mina <cpu_or_step128.mina> --out MINAKANUSHI-6.8B
```

For 27 GB use the GPU box. Laptop: probe mina or `--cards-only`. AutoConfig/AutoModel type tag. Never CausalLM. Never construct 6.8B.

## 100-episode walk

```text
python scripts/gate_v031_validate.py --root dataset/mina_6_8b_v03 --n 100
```
