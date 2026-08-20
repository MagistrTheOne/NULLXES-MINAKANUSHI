"""MinakanushiEngine — one external cognition cycle.

encode → position → update world → memory → uncertainty → situation
→ futures → strategies → constraints → authority → intent

Authority gates ActionPolicy. It does not disable WorldState or FutureEngine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from minakanushi.architecture.config import MinakanushiConfig
from minakanushi.architecture.mina_unit import pack_units
from minakanushi.architecture.model import MinakanushiSystem
from minakanushi.architecture.outputs import CycleTelemetry
from minakanushi.constraints.kernel import MinakanushiConstraintKernel
from minakanushi.future.engine import group_by_strategy
from minakanushi.identity.authority import AuthorityModel, AuthorityMode, candidate_from_intent
from minakanushi.identity.constants import SHORT_NAME
from minakanushi.identity.experience import ExperienceLog, ExperienceRecord, LESSON_POSITION, LESSON_VELOCITY
from minakanushi.focus.engine import FocusEngine, FocusState, focus_from_world
from minakanushi.identity.persona import PersonaModel
from minakanushi.identity.self_model import SelfModel
from minakanushi.memory.action_outcome import record_outcome
from minakanushi.memory.experience import ExperienceEngine
from minakanushi.perception.bridge import Observation
from minakanushi.policy.action_policy import ActionPolicy
from minakanushi.policy.intent import ActionIntent
from minakanushi.runtime.session import SessionState
from minakanushi.runtime.telemetry import LatencyClock, TelemetryLogger
from minakanushi.situation.core import SituationCore
from minakanushi.state.constructor import StateConstructor, empty_world_state
from minakanushi.strategy.engine import StrategyEngine
from minakanushi.training.checkpoint import load_mina, save_mina
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
        self.persona = PersonaModel()
        self.authority = AuthorityModel()
        self.experience = ExperienceEngine()
        self.focus_engine = FocusEngine()

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
        self_model = SelfModel.from_config(
            self.config.architecture.identity,
            self.config.architecture,
            hard_limits=tuple(self.config.simulation.hard_constraints),
        )
        return SessionState(
            cycle_id=0,
            episode_position=0.0,
            world=world,
            self_model=self_model,
            authority=AuthorityModel(mode=self.authority.mode, policy_enabled=self.authority.policy_enabled),
            persona=PersonaModel.from_dict(self.persona.to_dict()),
            focus=focus_from_world(world),
        )

    def set_mode(self, mode: AuthorityMode, *, policy_enabled: bool | None = None) -> None:
        self.authority.mode = mode
        if policy_enabled is not None:
            self.authority.policy_enabled = policy_enabled
            return
        if mode == AuthorityMode.AUTONOMOUS:
            self.authority.policy_enabled = True
        elif mode in {AuthorityMode.MANUAL, AuthorityMode.SAFE_HOLD}:
            self.authority.policy_enabled = False

    @torch.no_grad()
    def step(
        self,
        observations: Observation,
        state: SessionState,
        operator_intent: ActionIntent | None = None,
    ) -> EngineStep:
        clock = LatencyClock()
        arch = self.config.architecture
        authority = state.authority or self.authority
        self_model = state.self_model or SelfModel.from_config(arch.identity, arch)
        persona = state.persona or self.persona
        _ = persona.to_dict()
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
        exp_log = self_model.experience if self_model.experience is not None else ExperienceLog()
        xy_boost, vel_boost = self.experience.std_boost(state.world, exp_log)
        hints = self.system.memory.hints(state.world)
        constructed = self.constructor.apply(
            packed, state.world, fused, memory_hints=hints, experience_boost=(xy_boost, vel_boost)
        )
        _, core = self.system.observe_to_core(packed, constructed, hints)
        world = core.world_state
        self.system.memory.write(world, core.memory_write_candidates)
        if state.last_predicted is not None and state.last_intent is not None:
            outcome = record_outcome(
                intent=state.last_intent,
                belief_before=state.world,
                predicted=state.last_predicted,
                actual=world,
                event_time=float(observations.timestamp),
            )
            self_model.action_outcomes.append(outcome)
            if outcome.correction:
                lesson = LESSON_VELOCITY if "velocity" in outcome.lesson else LESSON_POSITION
                self_model.experience.append(
                    ExperienceRecord(
                        event_time=float(observations.timestamp),
                        situation="action_outcome",
                        action=state.last_intent.objective,
                        entity_id=1,
                        error_xy=outcome.prediction_error,
                        error_vel=outcome.prediction_error,
                        correction_required=True,
                        lesson=lesson,
                    )
                )
        prev_action = state.last_intent.objective if state.last_intent is not None else "OBSERVE"
        for record in self.experience.record_cycle(
            state.world, world, arch.dt, prev_action, float(observations.timestamp)
        ):
            self_model.experience.append(record)
        focus = self.focus_engine.select(world, self_model.experience, float(observations.timestamp))
        sit = self.situation.build(world, self.system.uncertainty(world, packed), (), focus=focus)
        candidates = list(self.strategy.generate(sit, self.config.simulation.home))
        if operator_intent is not None:
            extra = candidate_from_intent(operator_intent)
            if extra.strategy_id not in {c.strategy_id for c in candidates}:
                candidates.append(extra)
        futures = self.system.future.predict(world, candidates)
        by_id = group_by_strategy(futures)
        allowed, rejected, audits = self.constraints.filter(candidates, by_id)
        intent = authority.resolve(
            self.policy,
            allowed,
            by_id,
            self._goal(sit),
            observations.timestamp,
            operator_intent=operator_intent,
        )
        reasons = tuple(r for audit in audits if not audit.allowed for r in audit.reasons if "ok" not in r)
        self_model.authority_mode = authority.mode.value
        self_model.policy_enabled = authority.policy_enabled
        self_model.operator_connected = authority.operator_connected
        self_model.tick(arch.dt, sit.uncertainty, world.corrections)
        predicted = self.system.future.predict_belief(world, candidate_from_intent(intent), steps=1)
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
            extras={
                "authority_mode": authority.mode.value,
                "policy_enabled": authority.policy_enabled,
                "short_name": SHORT_NAME,
                "persona_bound": False,
                "experience_count": len(self_model.experience.records),
                "focus_type": focus.focus_type,
                "focus_target": focus.target_id,
                "action_outcomes": len(self_model.action_outcomes.records),
            },
        )
        self.telemetry.emit(telemetry)
        nxt = SessionState(
            cycle_id=state.cycle_id + 1,
            episode_position=state.episode_position + 1.0,
            world=world,
            last_intent=intent,
            causal=state.causal,
            self_model=self_model,
            authority=authority,
            persona=persona,
            focus=focus,
            last_predicted=predicted,
        )
        return EngineStep(state=nxt, action_intent=intent, telemetry=telemetry)

    def identity_bundle(self, state: SessionState) -> dict:
        return {
            "self_model": (state.self_model or SelfModel()).to_dict(),
            "authority": (state.authority or self.authority).to_dict(),
            "persona": (state.persona or self.persona).to_dict(),
            "focus": (state.focus.to_dict() if state.focus is not None else {}),
        }

    def save_checkpoint(self, path: str | Path, state: SessionState) -> Path:
        return save_mina(path, self.system, extras={"identity": self.identity_bundle(state)})

    def load_checkpoint(self, path: str | Path, state: SessionState) -> SessionState:
        manifest = load_mina(path, self.system)
        bundle = (manifest.get("train") or {}).get("identity") or {}
        if bundle.get("self_model"):
            state.self_model = SelfModel.from_dict(bundle["self_model"])
        if bundle.get("authority"):
            state.authority = AuthorityModel.from_dict(bundle["authority"])
            self.authority = state.authority
        if bundle.get("persona"):
            state.persona = PersonaModel.from_dict(bundle["persona"])
            self.persona = state.persona
        if bundle.get("focus"):
            state.focus = FocusState.from_dict(bundle["focus"])
        return state

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
