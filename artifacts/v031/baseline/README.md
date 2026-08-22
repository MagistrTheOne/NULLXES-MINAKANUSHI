# v0.3.1 baseline pack

Closed origin for the H200 experiment. Written by:

```text
python scripts/lock_v031_baseline.py --mina minakanushi_stage0_step128.mina --dataset dataset/mina_6_8b_v03 --out artifacts/v031/baseline
```

| File | Meaning |
|---|---|
| `checkpoint.sha256` | sha256 of step128 `*.mina`, or `MISSING` if the 27GB file is not on this machine |
| `metrics_before.json` | manifest metrics from that checkpoint (no 6.8B construct) |
| `capability_before.json` | Gate A–G measurements, or `NOT_RUN` |
| `reference_inference.pt` | cpu_dev snapshot only |
| `dataset_report.json` | extended v0.3 audit |
| `training_config.yaml` | frozen copy of `mina_6_8b_v03.yaml` |
| `git_commit.txt` | `git rev-parse HEAD` |
| `hardware.json` | H200 / FSDP2 / bf16 / 1000-step stop |

Do not invent a 6.8B forward on a laptop to fill `reference_inference.pt`.
