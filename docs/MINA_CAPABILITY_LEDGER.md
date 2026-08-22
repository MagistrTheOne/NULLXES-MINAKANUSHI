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

Update this table only from Gate A–F measurements. Do not update it from
`loss=` in a train log.

| Ability | Proven | Gate |
|---|---|---|
| World state reconstruction | yes (cpu_dev / Milestone 1) | v0.1 |
| Entity persistence | yes (cpu_dev / Milestone 1) | v0.1 |
| Belief revision | partial | Gate 03 |
| Memory improves future | yes on short cpu_dev probe; long-hide not yet | v0.3.1 / Gate E |
| Counterfactual futures | cpu_dev measured; 6.8B unknown | Gate D |
| Long horizon prediction | waiting | v0.4 |
| Causal attribution | protocol ready; 6.8B unknown | Gate C |
| Held-out vs seen | protocol ready; 6.8B unknown | Gate B |
| Revision honesty | protocol ready; `false=0` is not enough | Gate F |
| Multimodal grounding | no | Gate 9+ |

After B300/H200, fill **after** columns from the same protocol. If only train ADE
moves, write **memorization**, not **learned the world**.

## Protocol (cpu_dev now, 6.8B later)

Does not construct 6.8B on this machine.

```text
python scripts/gate_capability.py --out artifacts/v031/capability
```

Writes `reference_before/*.pt` (Gate A) and `capability_report.json`.

| Gate | Question | Fail looks like |
|---|---|---|
| A retention | Did new data smash old scenarios? | old ADE explodes, new ADE pretty |
| B held-out | Seed 7 vs seed 9999 | train ADE↓, unseen ADE flat |
| C causality | Why did the future change? | new picture → new answer, no `unexpected_physics` |
| D counterfactual | WAIT vs MOVE_TO | `future_distance ≈ 0` |
| E memory | Gone ~30 frames, then back | ADE(on) ≥ ADE(off) and reacquisition dead |
| F honesty | Revise when wrong, persist when right | `false_revision=0` because it never revises |

## Order before claiming training helped

```text
1. step128 freeze
2. export safetensors
3. v0.3.1 100-episode validation report
4. B300/H200 train
5. held-out evaluation (Gate B) + retention (Gate A)
6. update this ledger from numbers
7. then say "training improved prediction / revision / memory"
```

Step 7 is the first time the sentence may mention improvement. Not at step 4.
