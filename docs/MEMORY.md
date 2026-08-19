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
