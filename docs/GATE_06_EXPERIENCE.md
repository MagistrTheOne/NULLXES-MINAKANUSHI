# Gate 06 — Memory as Experience

**Status:** implemented on `cpu_dev`
**Does not:** Focus Engine (07), action-conditional world model (08),
`research_v01`, vision, language

Tensor memory (working ring + episodic slot store) stays. Gate 06 adds the
layer Maga named **experience**:

```text
Situation → Prediction → Reality → Error → Correction → Lesson
```

Not:

```text
store tensor
retrieve tensor
```

## Example

Predicted: object continues moving.
Reality: object stopped.
Error: velocity discontinuity.
Lesson: inflate unobserved `vel_std` for that entity.

The lesson changes later **belief distribution** for unobserved slots. It
never 50/50 with live evidence.

## Wiring

`ExperienceEngine.record_cycle(previous, updated, dt, action)` writes
`ExperienceRecord`s into `SelfModel.experience`.
`ExperienceEngine.std_boost` is applied in `StateConstructor` only on
persist (unobserved) slots.

Episodic tensor retrieval for `L_memory` is unchanged. Focus / curiosity
remain Gate 07.
