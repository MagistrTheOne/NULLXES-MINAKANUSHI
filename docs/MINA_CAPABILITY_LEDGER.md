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
| No shortcut | protocol ready; 6.8B unknown | Gate G |
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
