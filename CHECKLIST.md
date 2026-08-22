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
Корень Hub: `step64.mina` + `step128.mina` + **`step1128.mina` + safetensors зеркало step1128**  
Карточка: **research checkpoint, v0.3.1 verdict B, not accepted.** Коллекция: ASI (WorldModel).

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

### Сделано на H200

- [x] **Phase −1. Dataset pack** — CPU/2080 · seed 11 · `.READY_V031` · 900/100 · 32/64 · `pwm=false`
- [x] **Phase 0. Lock baseline** — verify read-only + `lock_v031_baseline.py` + step128
- [x] **Phase 1. Safetensors** — зеркало **step1128** (не step128) · `artifacts/v031/step1128/MINAKANUSHI-6.8B`
- [x] **Phase 2. CPU contract** — pack contract держался; C/E до train честно FAIL
- [x] **Phase 3. H200 1000 steps STOP** — `experiments/mina_6_8b_v031` · `minakanushi_stage0_step1128.mina`  
  Train-eval hint: **B / late C-signal**. Это не вердикт.

- [x] **HF publish** — `step1128.mina` + safetensors + metrics на [MINAKANUSHI-6.8B](https://huggingface.co/MagistrTheOne/MINAKANUSHI-6.8B), карточка *research / not accepted*, коллекция [ASI (WorldModel)](https://huggingface.co/collections/MagistrTheOne/asi-worldmodel-6a89f942152bb18dd68c144b), датасет [mina-6.8b-v03](https://huggingface.co/datasets/MagistrTheOne/mina-6.8b-v03)
- [x] **Compare ledger** — 100 heldout · step128 vs step1128 · `artifacts/v031/verdict/compare.json`  
  **Вариант B. v0.3.1 не accepted.**  
  ADE 6.78 → 0.845 · memory PASS (0.606 < 1.345 на n=40) · direction 0.41 → 0.64 · detection 0.85 → 0.64 · false_rev 0.05 ok · **C-signal: cf terminal diversity FAIL** (`cf≈0.000786`, std 3.4e-6). Action vector 1.414 доходит до Future Engine (‖ΔF‖≈0.40). Не ещё один train. Не v0.4.

### Вердикт (закрыт)

| | Что видим | Что делать |
|---|---|---|
| **A** | revision ↑ · memory ADE on < off · heldout ADE ↓ · false_revision ≈ 0 | v0.3.1 PASS → следующий цикл: geometry / long horizon |
| **B** ← факт | heldout ADE ↓ · memory PASS · occupied cf PASS · direction ↑ · detection ↓ (`sensor_delay`) | revision trigger calibration. **Не слои. Не ещё 1000 steps. Не v0.4.** |
| **C** | revision ломается · false_revision растёт | чинить causality в данных/лоссе. **Не слои.** |

---

## v0.3.2 diagnostic-fix (закрыт, не train)

План: `docs/V032_DIAGNOSTIC.md`

- [x] `python scripts/diagnose_counterfactual_v031.py` — CPU **fork A**: official cf = agent/512 · recovered Δ ≈ 0.402 · recovered std ≈ 0.0017
- [x] sampler replay 129..1128 — WAIT/MOVE 1000/1000; sensor_delay 45; gone_forever 49. Sampler не bottleneck
- [x] embodiment: `sensor_delay` detect **0.00** (n=13); correction L1–L3 / conflict detect 1.0. gone_forever n=3 не трогать global loss
- [x] occupied heldout-100 **без обучения** — existence/diversity PASS; variant всё ещё **B** (detection 0.85 → 0.64)

## v0.3.1-R Revision Trigger Diagnostic (не train)

План: `docs/V031R_REVISION.md`

Оставшийся дефект — не counterfactual и не memory. Калибровка кнопки «пересмотреть belief», особенно `sensor_delay`.

- [x] CPU forensic: delay = timestamp; train frame = `length//2` без mover; one-step и delay-path < `0.25`; empty teacher → detected=0
- [ ] H200 live dump `max_before_d` step128 vs step1128 **только sensor_delay heldout** (`scripts/diagnose_revision_v031r.py`)
- [ ] локальный патч teacher/метрики — только после live cut-point. **Не общий train. Не v0.4.**

## Не этот цикл

Длинный 6.8B, pixels → MinaUnit, органы, другие тела — не этот отчёт. v0.4 (geometry) только после A. Сейчас B: не открывать параллельный train.

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
