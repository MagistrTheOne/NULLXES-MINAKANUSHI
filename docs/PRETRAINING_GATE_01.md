# Pre-training validation gate 01

Static source audit. Nothing in this file was measured by executing the model
on the authoring workstation.

## Forward graph (implemented)

| Boundary | Input | Output | Grad |
|---|---|---|---|
| SyntheticWorld.observe | world bodies | Observation (python) | no |
| PerceptionBridge.encode | Observation | list[MinaUnit], embedding [D] | yes (encoders) |
| pack_units | list[MinaUnit] | MinaUnitBatch B=1, N=max_obs | yes through embeddings |
| NPF | event_time, arrival_time, source_rate, seq, space, episode, memory_age, source_id | PositionState [B,N,D] | yes |
| StateConstructor | batch + previous WorldState | WorldState (ID match discrete) | yes through latents/xy copies |
| DWC | world [B,Nw,D], obs [B,No,D], memory [B,Nw,D] | CoreOutput | yes |
| MemoryEngine.hints | world, optional live_writes [B,Nw,D] | hints [B,Nw,D] | yes if live_writes |
| UncertaintyEngine | latent [B,Nw,D] | channels [B,Nw,U] | yes (proj) |
| FutureEngine | cloned world, strategies | K branches / strategy, xy [H,Nw,2] | yes (residual, logits, unc) |
| ConstraintKernel | StrategyCandidate + trajectories | AllowedStrategy only | no |
| ActionPolicy | AllowedStrategy | ActionIntent | no |

Batch semantics of the runtime loop: B=1. Trainer currently B=1 overfit episodes.
Time semantics: timestamp=event_time; arrival_time may differ; memory_age=now-event_time.

## Parameter estimates (formulas, not numel)

world_slots and memory_slots are state/buffers, not parameters.
prediction horizon is an unroll length, not a parameter multiplier.

Dominant term: DWC ≈ core_depth × 12 × latent_dim²

| Profile | ESTIMATE total | Notes |
|---|---:|---|
| cpu_dev D=64 L=2 | ~2.1e5 | intended validation size |
| gpu_train_v01 D=256 L=6 | ~5.5e6 | intermediate |
| research_v01 D=2048 L=24 | ~1.3e9 | large because 24×12×2048², not because of 512 slots |

research_v01 is large by width×depth design. It is not an accidental slot explosion.
Do not instantiate research_v01 on the CPU validation machine.

## Dead objectives found and corrected

| Loss | Before | After |
|---|---|---|
| L_causal | pred.vel vs pred.vel.detach() → 0 | pred.vel vs GT vel |
| L_memory | duplicate of L_state; zeros memory | occluded-slot error through live_writes |
| L_action | strategies only; P from softmax(-U) | strategy split + intra-strategy branch margin |
| L_future | one traj / strategy | still trains residual; now K branches |
| L_state, L_temporal, L_uncertainty, L_repr | live | live |

## Firewalls

- FutureEngine clones WorldState and asserts data_ptr identity after predict.
- P is softmax over branch logits within one strategy. U is softplus of a separate head.
- ActionPolicy.select requires AllowedStrategy minted only by the kernel.

## H100/H200

A–J require execution. Status: NOT EXECUTED. Do not scale to H100/H200 until A–J pass on the authorized machine.
