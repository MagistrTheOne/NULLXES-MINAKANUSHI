# MINA Capability Ledger

This document is a brake on storytelling. A lower loss is not a new ability.
A checkpoint that trains is not a mind.

```text
FORBIDDEN
  MINA получила сознание
  MINA поняла мир
  MINA стала AGI
  модель думает как человек

ALLOWED
  MINA улучшила prediction error
  MINA научилась корректировать belief
  memory улучшает future prediction
  ActionIntent выбирается с учётом world state
```

Update this table only from Gate A–G measurements. Do not update it from
`loss=` in a train log. A PASS row must cite `n=` and the numbers, for example:

```text
Memory improves future
PASS
heldout hide scenario:
ADE ON 1.8
ADE OFF 3.4
n=500
```

Gate C `pass` is `revision_detected > 0` on `unexpected_stop`, not a hardcoded
event label. Gate E `pass` is ADE(on) < ADE(off), not “metric is not None”.

| Ability | Proven | Gate |
|---|---|---|
| World state reconstruction | yes (cpu_dev / Milestone 1) | v0.1 |
| Entity persistence | yes (cpu_dev / Milestone 1) | v0.1 |
| Belief revision | partial (Gate 03 hidden_correction); unexpected_stop not proven | Gate 03 / C |
| Memory improves future | no at Gate E length=32 (ADE on worse than off). Short probe is not this gate. | v0.3.1 / Gate E |
| Counterfactual existence | yes on cpu_dev (WAIT vs MOVE_TO distance > 0); 6.8B unknown | Gate D |
| Counterfactual diversity | NOT PASS. Pack min=max≈0.779, std≈0. One arena geometry. v0.4 geometry expansion. | v0.3 audit / v0.4 |
| Long horizon prediction | waiting | v0.4 |
| Causal attribution | no; revision_detected=0, picture_in_picture_out on unexpected_stop | Gate C |
| Held-out vs seen | protocol ready; 6.8B unknown | Gate B |
| Revision honesty | protocol ready; `false=0` is not enough | Gate F |
| No shortcut | protocol ready; 6.8B unknown | Gate G |
| Multimodal grounding | no | Gate 9+ |

```text
Counterfactual:
PASS for existence
NOT PASS for diversity
```

After B300/H200, fill **after** columns from the same protocol. If only train ADE
moves, write **memorization**, not **learned the world**. Compare
`capability_before` vs `capability_after`, not step128 loss vs stepN loss.

## Protocol (cpu_dev now, 6.8B later)

Does not construct 6.8B on this machine.

```text
python scripts/gate_capability.py --out artifacts/v031/capability
```

Writes `reference_before/*.pt` (Gate A) and `capability_report.json`.
Unproven gates are printed. That is not a license to mark PASS.

| Gate | Question | Fail looks like |
|---|---|---|
| A retention | Did new data smash old scenarios? | old ADE explodes, new ADE pretty |
| B held-out | Seed 7 vs seed 9999 | train ADE↓, unseen ADE flat |
| C causality | Why did the future change? | new picture → new answer, `revision_detected=0` |
| D counterfactual | WAIT vs MOVE_TO | `future_distance ≈ 0` |
| E memory | Gone ~30 frames, then back | ADE(on) ≥ ADE(off) |
| F honesty | Revise when wrong, persist when right | `false_revision=0` because it never revises |
| G no shortcut | Ablate vision / delay telemetry; permute speed·position | both channels ignored, or permute is a no-op |

## Order before claiming training helped

```text
1. v0.3.1 baseline pack     scripts/lock_v031_baseline.py → artifacts/v031/baseline
2. held-out split           scripts/split_heldout.py  (seed, scenario, episode_index)
3. extended audit           scripts/audit_curriculum.py --gate
4. export safetensors       scripts/export_safetensors.py
5. H200 Phase 1             1000 steps then STOP
6. retention + held-out     scripts/compare_v031.py
7. update this ledger from numbers
8. then say "training improved prediction / revision / memory"
```

Step 8 is the first time the sentence may mention improvement. Not at step 5.
