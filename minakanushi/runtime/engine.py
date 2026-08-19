"""MinakanushiEngine — one external cognition cycle.

encode → position → update world → memory → uncertainty → situation
→ futures → strategies → constraints → policy → intent
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from minakanushi.architecture.config import MinakanushiConfig
from minakanushi.architecture.mina_unit import pack_units
from minakanushi.architecture.model import MinakanushiSystem
from minakanushi.architecture.outputs import CycleTelemetry
from minakanushi.constraints.kernel import MinakanushiConstraintKernel
from minakanushi.future.engine import group_by_strategy
from minakanushi.perception.bridge import Observation
from minakanushi.policy.action_policy import ActionPolicy
from minakanushi.policy.intent import ActionIntent
from minakanushi.runtime.session import SessionState
from minakanushi.runtime.telemetry import LatencyClock, TelemetryLogger
from minakanushi.situation.core import SituationCore
from minakanushi.state.constructor import StateConstructor, empty_world_state
from minakanushi.strategy.engine import StrategyEngine
from minakanushi.utils.seed import seed_everything
from minakanushi.utils.tensors import resolve_device, resolve_dtype


@dataclass
class EngineStep:
    state: SessionState
    action_intent: ActionIntent
    telemetry: CycleTelemetry


class MinakanushiEngine:
    def __init__(self, config: MinakanushiConfig) -> None:
        self.config = config
        seed_everything(config.runtime.seed)
        self.device = resolve_device(config.runtime.device)
        self.dtype = resolve_dtype("float32")
        self.system = MinakanushiSystem(config.architecture).to(self.device)
        self.constructor = StateConstructor(config.architecture)
        self.constraints = MinakanushiConstraintKernel(config.simulation)
        self.strategy = StrategyEngine()
        self.policy = ActionPolicy()
        self.situation = SituationCore()
        self.telemetry = TelemetryLogger(config.runtime.log_level)

    @property
    def future(self):
        return self.system.future

    @property
    def memory(self):
        return self.system.memory

    def initialize(self) -> SessionState:
        world = empty_world_state(
            self.config.architecture,
            1,
            device=self.device,
            dtype=self.dtype,
            timestamp=0.0,
        )
        start = self.config.simulation.agent_start
        world.entity_xy[0, 0, 0] = start[0]
        world.entity_xy[0, 0, 1] = start[1]
        world.confidence[0, 0] = 1.0
        return SessionState(cycle_id=0, episode_position=0.0, world=world)

    @torch.no_grad()
    def step(self, observations: Observation, state: SessionState) -> EngineStep:
        clock = LatencyClock()
        arch = self.config.architecture
        units_list = self.system.perception.encode(observations, device=self.device, dtype=self.dtype)
        packed = pack_units(
            units_list,
            batch_index=0,
            max_units=arch.max_observations,
            latent_dim=arch.latent_dim,
            episode_position=state.episode_position,
            now=observations.arrival_time if observations.arrival_time is not None else observations.timestamp,
            device=self.device,
            dtype=self.dtype,
        )
        positioned = self.system.position_units(packed)
        fused = packed.semantic_embedding + positioned.embedding
        hints = self.system.memory.hints(state.world)
        constructed = self.constructor.apply(packed, state.world, fused, memory_hints=hints)
        _, core = self.system.observe_to_core(packed, constructed, hints)
        world = core.world_state
        self.system.memory.write(world, core.memory_write_candidates)
        sit = self.situation.build(world, self.system.uncertainty(world, packed), ())
        candidates = self.strategy.generate(sit, self.config.simulation.home)
        futures = self.system.future.predict(world, candidates)
        by_id = group_by_strategy(futures)
        allowed, rejected, audits = self.constraints.filter(candidates, by_id)
        intent = self.policy.select(allowed, by_id, self._goal(sit), observations.timestamp)
        reasons = tuple(r for audit in audits if not audit.allowed for r in audit.reasons if "ok" not in r)
        telemetry = CycleTelemetry(
            cycle_id=state.cycle_id + 1,
            physical_time=observations.timestamp,
            observation_count=len(units_list),
            entity_count=world.entity_count,
            event_count=len(state.causal.events),
            world_state_confidence=float(world.confidence[0, world.occupied[0]].mean().item()) if bool(world.occupied.any()) else 0.0,
            uncertainty=sit.uncertainty,
            memory_reads=self.system.memory.episodic.reads,
            memory_writes=self.system.memory.episodic.writes,
            future_branches=len(futures),
            candidate_strategies=len(candidates),
            rejected_strategies=len(rejected),
            rejection_reasons=reasons,
            selected_strategy=intent.strategy_id,
            cognition_cycles=core.cognition_cycles,
            latency_ms=clock.ms(),
        )
        self.telemetry.emit(telemetry)
        nxt = SessionState(
            cycle_id=state.cycle_id + 1,
            episode_position=state.episode_position + 1.0,
            world=world,
            last_intent=intent,
            causal=state.causal,
        )
        return EngineStep(state=nxt, action_intent=intent, telemetry=telemetry)

    def _goal(self, sit) -> tuple[float, float]:
        world = sit.world_state
        from minakanushi.architecture.mina_unit import KIND_IDS

        for slot in world.occupied[0].nonzero(as_tuple=False).flatten().tolist():
            if int(world.kind[0, slot].item()) == KIND_IDS["target"]:
                return (
                    float(world.entity_xy[0, slot, 0].item()),
                    float(world.entity_xy[0, slot, 1].item()),
                )
        return self.config.simulation.home
