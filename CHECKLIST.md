# Чеклист этапов — NULLXES MINAKANUSHI

**Старт плана:** 21-08-2026  
**Freeze:** `7aba976` · профиль `minakanushi_6_8b` (6 799 130 646)  
**Не делать:** слои/DWC/MoE/language head · train 6.8B на CPU / RTX PRO 6000 / RTX 2080 / 1× H100 80GB

Метрики ниже — из закрытых прогонов. Пустая ячейка = ещё не мерили на этом этапе.

---

## Машины

| Роль | Железо | Что на ней | Что нельзя |
|---|---|---|---|
| Ноут / CPU / RTX 2080 | Windows | тесты, JSON v0.3, CPU acceptance, freeze YAML | не качать 27GB `.mina`, не train 6.8B |
| **Финальный этап v0.3.1** | **1× H200 SXM 141 GB** | download step128 с HF → lock → safetensors → 1000 steps STOP → compare | не B300 для этого эксперимента, не 2080 |
| Stage A (закрыт) | 1× RTX PRO 6000 BW 96 GB | только `gpu_train_v01` 6.2M, Gate 03B n=1000 | train 6.8B |
| Status Core v0.1 (закрыт) | 1× H200 SXM 141 GB | 6.8B FSDP2 bf16, step64 | не train на 6000 |
| Status Core v0.2 (закрыт) | **1× B300 ~288 GB** | resume IdentityBound + JSON `dataset/mina_6_8b`, steps 65–128 | сырой HF video / Cosmos / LeRobot RGB |
| Infer (не train) | 6000 BW или 1× H100 80GB | веса bf16 ~13.6 GB + голова мира | не путать с train 6.8B |

HF артефакт: [MagistrTheOne/MINAKANUSHI-6.8B](https://huggingface.co/MagistrTheOne/MINAKANUSHI-6.8B)  
Корень Hub: `minakanushi_stage0_step64.mina` + `minakanushi_stage0_step128.mina`

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

## Закрыто с 21-08-2026 — Status Core v0.2

Пункты 0–2 с чеклиста 21-го **закрыты**. Провал гейта → чинить данные/resume, **не** добавлять слои.

- [x] **0. Identity Initialization** — CPU, без construct 6.8B  
  `step64.mina` → штамп паспорта → IdentityBound. Нет `identity_loss`.  
  `python scripts/identity_init.py`

- [x] **1. JSON curriculum 1000 + фильтр** — CPU  
  SOURCE OF TRUTH: `dataset/mina_6_8b` (наш генератор, не Hub dump).  
  `--n 250` × 4 фазы (physics → agency → causality → embodiment).  
  Audit: ключи 6.8B, `pwm=false`. Minari/D4RL/Open-X — adapter, не raw.  
  Отвергнуто: NVIDIA PhysicalAI / Cosmos video, LeRobot RGB (pixels = не этот цикл).

- [x] **2. Resume v0.2** — **1× B300** · git `ede6bda`  
  тот же модель: optimizer + RNG + cursor + scheduler + identity, не clone.  
  `dataset/mina_6_8b` в лоссе. steps **65–128**.  
  Артефакт: `minakanushi_stage0_step128.mina` на HF.

  | метрика | значение | кратко |
  |---|---|---|
  | loss | 41.10 | лог, не гейт |
  | step time (steady) | fwd ~1.42 s · bwd ~0.58 s | B300 |
  | future ADE / FDE | 2.05 / 0.68 | лучше v0.1, не PASS интеллекта |
  | world position error | 0.55 | |
  | uncertainty calibration | 0.19 | |
  | persistence / reacquisition | **1.0 / 1.0** | |
  | constraint_violation_count | **0** | |
  | closed_loop_success_rate | **1.0** | |
  | false_revision_rate | **0.0** | |
  | revision_accuracy | **0.0** | не PASS |
  | branch_coverage | **0.0** | не PASS |

  Loss на mixed JSON — не архитектурный гейт. Revision/branch — ещё не победа. Слои не трогать.

- [x] **Контур v0.3.1 (код, не H200 job)** — freeze YAML, resume-аудит, sampler, hidden-correction 1–3, cpu_dev memory ADE, safetensors roundtrip, capability ledger с честными C/E, `lock_v031_baseline.py`, `check_freeze.py`, `gate_v031_acceptance.py`.  
  Ledger-баги закрыты: Gate C больше не зелёный от имени события; Gate E больше не зелёный от «вектор есть».  
  `python scripts/check_freeze.py` на ноуте: **PASS** (latent 4096 / depth 32 / 6 799 130 646). 6.8B не строится.

Снято с этого цикла (не гейт, не блокер H200):

- ~~Acceptance Gate v0.2 как дверь наружу~~ — живой контракт: `scripts/gate_v031_acceptance.py` (FAIL = FAIL).
- ~~Yunmu review / dry-run / humanoid package~~ — не этап MINA. Контроллер снаружи, если когда-нибудь понадобится, не из этого чеклиста.
- ~~HF safetensors «после Acceptance Gate»~~ — перенесено в финальный этап на H200 (step128, не step64).

---

## Финальный этап — v0.3.1 на 1× H200

Один эксперимент. Не «крутим модельку». Не B300. Не 2080.

Контракт: `docs/MINA_TRAINING_CONTRACT_v03.md`  
Конфиг: `configs/training/mina_6_8b_v03.yaml` (`steps: 1000`, `eval_every: 50`, `checkpoint_every: 250`, `dataset_split: train`)

```text
RTX 2080 / CPU
  prepare_v031_dataset  →  .READY_V031
        ↓ COPY pack
H200
  verify only
        ↓
  lock baseline + step128
        ↓
  export safetensors
        ↓
  1000 STOP
        ↓
  compare ledger
        ↓
  вердикт A / B / C
```

### Ещё открыто (только это)

- [ ] **Phase −1. Dataset pack** — ноут / 2080, не H200  
  `python scripts/prepare_v031_dataset.py --root dataset/mina_6_8b_v03 --n 250 --seed 11`  
  Default `--n=250`, `--profile v031`. `n<250` на production — FAIL. Dev: `--profile cpu_dev`.  
  После PASS: `dataset_manifest.json` + `.READY_V031`. Копировать **всю папку** на H200.  
  `train.py` / `mina_6_8b_v03.yaml` без marker — refuse. H200 split не дописывает.

- [ ] **Phase 0. Lock baseline** — на H200, после `hf download` и copy pack  
  `python scripts/verify_v031_dataset.py --root dataset/mina_6_8b_v03`  
  (read-only: не generate, не split)  
  `python scripts/lock_v031_baseline.py --mina minakanushi_stage0_step128.mina --require-mina --out artifacts/v031/baseline`  
  `python scripts/check_freeze.py --checkpoint minakanushi_stage0_step128.mina`  
  Точка невозврата: `checkpoint.sha256` + metrics + capability_before + dataset_report + training_config + hardware + git_commit.

- [ ] **Phase 1. Safetensors** — там же, не локально  
  `python scripts/export_safetensors.py --mina minakanushi_stage0_step128.mina --out MINAKANUSHI-6.8B`  
  `python scripts/test_hf_reload.py --path MINAKANUSHI-6.8B`  
  Гейт: load tensor · shape match · **6799130646** · AutoModel type tag · **AutoModelForCausalLM absent**.

- [ ] **Phase 2. CPU contract** — ноут/2080 после prepare, не обучение  
  `python scripts/gate_v031_acceptance.py --dataset dataset/mina_6_8b_v03 --split heldout`  
  Без `.READY_V031` — FAIL, split не дописывается.  
  Ожидание до H200: C/E **честно FAIL** (`revision_detected = 0`, ADE memory on ≮ off). Это baseline, не сломанный скрипт.

- [ ] **Phase 3. H200 1000 steps STOP**  
  ```text
  torchrun --nproc_per_node=1 scripts/train.py \
    --config configs/training/mina_6_8b_v03.yaml \
    --resume minakanushi_stage0_step128.mina \
    --out experiments/mina_v031_h200
  ```  
  Смотреть каждые 50: `revision_detected` · `false_revision` · ADE(memory on) < ADE(memory off) · `std(distance)` · **heldout ADE**.  
  Не смотреть `loss ↓` как победу. После 1000 — стоп, даже если хочется ещё 5000.

- [ ] **Compare ledger**  
  `python scripts/compare_v031.py --before artifacts/v031/baseline/capability_before.json --after experiments/mina_v031_h200/capability_after.json`

### Вердикт (после compare, не раньше)

| | Что видим | Что делать |
|---|---|---|
| **A** | revision ↑ · memory ADE on < off · heldout ADE ↓ · false_revision ≈ 0 | v0.3.1 PASS → следующий цикл: geometry / long horizon |
| **B** | loss ↓ · train ADE ↓ · heldout = · memory = · revision = | подгонка. Менять sampler/curriculum/scenarios. **Не архитектуру.** |
| **C** | revision ломается · false_revision растёт | чинить causality в данных/лоссе. **Не слои.** |

---

## Не этот цикл

Длинный 6.8B, pixels → MinaUnit, органы, другие тела — **после** вердикта A/B/C. Не смешивать с Phase 0–3. Не открывать параллельный train.

---

## Запрет на каждом этапе

```text
не менять latent_dim / core_depth / world_slots / memory_slots
не подменять DWC, не language head, не identity_loss
не train authority как neural objective
не AutoModelForCausalLM
6000 BW ≠ машина 6.8B train
RTX 2080 ≠ машина для 27GB .mina
ActionIntent ≠ PWM
```
