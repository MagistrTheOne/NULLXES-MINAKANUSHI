# Memory

Memory is native cognitive state, not document RAG.

## Working

Short ring buffer of pooled world embeddings. High-resolution current context.

## Episodic

Slot-aligned store of `(entity_id, xy, embedding, importance)` written each
cycle. Retrieval is by entity identity, then mixed into unobserved occupied
slots. Semantic consolidation is not in the Milestone 1 runtime.

Relevance uses identity, recency (via NPF memory age), occupancy, and
confidence — not cosine search over text chunks.

## Belief is not memory

```text
PAST      → Memory     what was
CURRENT   → Belief     what is probable now
POSSIBLE  → Future     what may be
```

Memory is a prior for unobserved slots (latent hints only). It does not
average with live evidence on position. Current kinematics live in WorldState
belief. Futures must not write back into belief. See
`docs/GATE_03A_BELIEF_REVISION.md`.
