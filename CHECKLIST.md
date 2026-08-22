# Чеклист этапов — NULLXES MINAKANUSHI

**Старт плана:** 21-08-2026  
**Freeze:** `7aba976` · профиль `minakanushi_6_8b` (6 799 130 646)  
**Не делать:** слои/DWC/MoE/language head · train 6.8B на CPU / RTX PRO 6000 / 1× H100 80GB

Метрики ниже — из закрытых прогонов. Пустая ячейка = ещё не мерили на этом этапе.

---

## Машины

| Роль | Железо | Что на ней | Что нельзя |
|---|---|---|---|
| Ноут / CPU | Windows CPU | тесты, v0.2 gate (`cpu_dev`), Identity Init (штамп zip), JSON-генератор, пакет V2 MM | конструировать 6.8B |
| Stage A (закрыт) | 1× RTX PRO 6000 BW 96 GB · pod `gn3eqwxuht23qs` · ~$2.09–2.40/ч | только `gpu_train_v01` 6.2M, Gate 03B n=1000 | train 6.8B |
| Status Core / sanity | **1× H200 SXM 141 GB** · ~$4.59/ч + диск | 6.8B FSDP2 bf16, step64; запас если B300 ещё нет | не train на 6000 |
| След. неделя (цель) | **1× B300 ~288 GB** | v0.2 resume + обучение на `dataset/mina_6_8b` | сырой HF video / Cosmos / LeRobot RGB |
| Длинный train | **2× H200** или тот же **1× B300** | полный AdamW 6.8B | 1× H100 80GB train |
| Infer / Yunmu dry-run | 6000 BW или 1× H100 80GB | веса bf16 ~13.6 GB + голова мира | не путать с train |

HF артефакт: [MagistrTheOne/MINAKANUSHI-6.8B](https://huggingface.co/MagistrTheOne/MINAKANUSHI-6.8B) · корень репы: `minakanushi_stage0_step64.mina` + `minakanushi_stage0_step128.mina`

## Бюджет B300 (MVP обученной машины)

Тариф: **$7.89 / GPU·ч**. Диск в сумму не входит (меняется). Степ-тайм 6.8B на B300 в репо **не замерян** (H200 step64 был, wall-clock в карточку не записали). Ниже — потолки, не обещание часов.

| Конверт | GPU·ч | $ | Что внутри |
|---|---:|---:|---|
| Минимум | ~8 | **~$65** | подъём + один job `steps: 64` если шаг ~2–4 мин |
| **MVP потолок (заложить)** | **~32** | **$250** | подъём + 64 шага + один рестарт/OOM + eval + 4–8 ч буфер |
| Жёсткий стоп | ~50 | **$400** | второй сегмент 64 или шаг оказался ~8–10 мин |
| Ошибка «забыл Stop» | 24 / сут | **$189 / сут** | неделя Running ≈ **$1320** — это не MVP |

**Заложить на MVP: $250. Резать под $400. Не держать Running без job.**

CPU (IdentityBound + `--n 250` JSON + audit) = $0 GPU. На B300 только resume с `dataset_root`.

---

## Закрыто до 21-08-2026

- [x] **Gate 09 Runtime** — CPU · `cpu_dev` · цикл `observe→intent→restore` принят
- [x] **Stage A GPU** — 6000 BW · 6.2M · CUDA/bf16/AMP/`.mina` (не интеллект, стек)
- [x] **Gate 03B hidden direction** — 6000 BW · `gpu_train_v01` · n=1000 · SHA `7fd8ef6` · 2026-08-20

  hidden_correction: direction mean **0.76** / median **1.0**, detected **0.986**, false_revision **0.0**  
  conflict: direction mean **0.92**, detected **0.94**, false_revision **0.0**  
  *пояснение:* ревизия ловится, ложных почти нет; направление — вопрос данных, не повод менять архитектуру.

- [x] **6.8B Status Core v0.1 step 64** — 1× H200 · FSDP2 ZeRO-3 bf16 · seed 11 · git `d70bfc0`

  | метрика | значение | кратко |
  |---|---|---|
  | loss | 78.36 | лог, не гейт архитектуры |
  | future ADE / FDE | 3.42 / 0.81 | мир ещё грубый |
  | world position error | 1.07 | |
  | uncertainty calibration | 0.38 | |
  | persistence / reacquisition | **1.0 / 1.0** | сущности не пропадают зря |
  | constraint_violation_count | **0** | хард-зоны не покупаются score |
  | closed_loop_success_rate | **1.0** | петля сима закрыта |
  | false_revision_rate | **0.0** | |

  *Не PASS обучения:* JSON не в лоссе, не было `--resume`, revision после step 1 пустой, branch coverage ≈ 0, memory был L2 латента.

---

## С 21-08-2026 — Status Core v0.2 (pipeline + data)

Порядок жёсткий. Провал гейта → чинить данные/resume, **не** добавлять слои.

- [ ] **0. Identity Initialization** — CPU (без construct 6.8B)  
  `step64.mina` → штамп паспорта → `MINA-6.8B-IdentityBound.mina`  
  *не train, нет `identity_loss`.*  
  `python scripts/identity_init.py --checkpoint … --out …/MINA-6.8B-IdentityBound.mina`

- [ ] **1. JSON curriculum 1000 + фильтр** — CPU  
  SOURCE OF TRUTH: `dataset/mina_6_8b` (наш генератор, не Hub dump).  
  `--n 250` × 4 фазы (physics → agency → causality → embodiment), off git.  
  Фильтр качества: `scripts/audit_curriculum.py` — ключи 6.8B, `pwm=false`, фазы, transitions.  
  Фильтр источника: в лосс только native JSON; Minari/D4RL/Open-X — **adapter** в Observation/ActionIntent, не raw.  
  Отвергнуть: NVIDIA PhysicalAI / Cosmos video, LeRobot RGB (pixels = Gate 9+, не v0.2).  
  `python scripts/generate_6_8b_curriculum.py --root dataset/mina_6_8b --n 250`  
  `python scripts/audit_curriculum.py`

- [ ] **2. Resume v0.2** — **1× B300** (след. неделя; иначе 1× H200)  
  тот же модель: optimizer + RNG + cursor + scheduler + identity, не clone  
  `dataset_root: dataset/mina_6_8b` в `mina_6_8b_v02.yaml`  
  `torchrun … scripts/train.py --config configs/training/mina_6_8b_v02.yaml --resume IdentityBound.mina`

- [ ] **3. Acceptance Gate** — сначала CPU `cpu_dev`  
  `python scripts/gate_v02_acceptance.py`  
  предсказать · поймать ложный belief · revise · remember (`memory_future_delta`) · другой future · authority  
  метрики ядра: ADE/FDE/uncertainty · revision_accuracy/latency/false_revision · memory_future_delta · future_diversity · counterfactual_quality

- [ ] **4. Yunmu review** — пакет: IdentityBound + доки + лимиты  
  контроллер снаружи, ActionIntent внутрь, не PWM. **Не** открывать, пока п.3 не PASS.

- [ ] **HF safetensors mirror** — после Acceptance Gate, не step64.  
  `.mina` = канон. safetensors = витрина Hub.  
  `python scripts/export_safetensors.py --mina final.mina --out hf_mirror`

- [ ] **Curriculum v0.3** — `dataset/mina_6_8b_v03` · 32/64 кадров · correction density · future forks  
  `python scripts/generate_6_8b_curriculum.py --root dataset/mina_6_8b_v03 --n 250`  
  `python scripts/split_heldout.py --root dataset/mina_6_8b_v03`  
  `python scripts/audit_curriculum.py --root dataset/mina_6_8b_v03 --gate`  
  H200 resume только после этого гейта.

- [ ] **Optimization Pass v0.3.1** — контур, не сеть. `docs/MINA_OPTIMIZATION_V031.md`  
  `python scripts/freeze_step128.py --mina minakanushi_stage0_step128.mina --out artifacts/v031/step128`  
  `python scripts/audit_resume.py --mina minakanushi_stage0_step128.mina`  
  `python scripts/gate_v031_export.py --mina probe.mina --out artifacts/v031/hf_probe`  
  `python scripts/gate_v031_validate.py --root dataset/mina_6_8b_v03 --n 100`  
  `python scripts/gate_v031_loss_probe.py --steps 32`  
  `python scripts/gate_capability.py --out artifacts/v031/capability`

- [ ] **Pre-Training Lock v0.3.1** — чистый эксперимент, не «умнее». `docs/MINA_TRAINING_CONTRACT_v03.md`  
  `python scripts/lock_v031_baseline.py --mina minakanushi_stage0_step128.mina --dataset dataset/mina_6_8b_v03 --out artifacts/v031/baseline`  
  Phase 1 на H200: **1000 steps then STOP**. Смотреть loss / ADE/FDE / revision / memory_future_delta / counterfactual.  
  После: `python scripts/compare_v031.py --before artifacts/v031/baseline/capability_before.json --after artifacts/v031/after/capability_report.json`  
  Ledger только из цифр (`n=`). Gate G = no shortcut. Не стартовать H200 без baseline pack + held-out + audit.

---

## Позже (не смешивать с 1.0)

- [ ] **Длинный 6.8B** — тот же **1× B300** (или 2× H200) · тот же freeze · тот же `mina_6_8b` JSON
- [ ] **Gate 9+ perception** — pixels → MinaUnit
- [ ] **MINA V2 MM** — эксперимент уже в `models/MINA-V2-MM/` + `docs/experiments/MINA_V2_MULTIMODAL.md`  
  органы → MinaUnit, не VLA/Cosmos; **не исполнять как train**, пока Yunmu/Gate 9+ не закрыты
- [ ] **MINA 3.0** — одно cognition, разные тела (humanoid / UAV / vehicle)

---

## Запрет на каждом этапе

```text
не менять latent_dim / core_depth / world_slots / memory_slots
не подменять DWC, не language head, не identity_loss
не train authority как neural objective
6000 BW ≠ машина 6.8B train
```
