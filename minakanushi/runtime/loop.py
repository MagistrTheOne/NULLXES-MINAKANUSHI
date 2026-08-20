"""MinakanushiRuntime — continuous process. Not Engine. Not a personality.

Engine.step() is one cognition tick.
Runtime.cycle() owns time: observe → cognize → authorize → execute → remember.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from minakanushi.architecture.config import MinakanushiConfig
from minakanushi.architecture.outputs import CycleTelemetry
from minakanushi.identity.authority import AuthorityMode
from minakanushi.perception.bridge import Observation
from minakanushi.policy.intent import ActionIntent
from minakanushi.runtime.engine import EngineStep, MinakanushiEngine
from minakanushi.runtime.metrics import RuntimeMetrics
from minakanushi.runtime.platform import ActionResult, SyntheticPlatform
from minakanushi.runtime.session import SessionState
from minakanushi.runtime.snapshot import belief_fingerprint, dump_world, load_world
from minakanushi.runtime.state import RuntimeState
from minakanushi.training.checkpoint import load_mina, save_mina
from simulations.synthetic_world.world import SyntheticWorld


@dataclass
class CycleResult:
    runtime: RuntimeState
    session: SessionState
    action_intent: ActionIntent
    action_result: ActionResult
    telemetry: CycleTelemetry
    metrics: RuntimeMetrics


class MinakanushiRuntime:
    def __init__(
        self,
        config: MinakanushiConfig,
        *,
        seed: int | None = None,
        platform: SyntheticPlatform | None = None,
        engine: MinakanushiEngine | None = None,
    ) -> None:
        self.engine = engine or MinakanushiEngine(config)
        world = SyntheticWorld(config.simulation, seed=config.runtime.seed if seed is None else seed)
        self.platform = platform or SyntheticPlatform(world)
        self.session = self.engine.initialize()
        self.session.authority = self.engine.authority
        self.state = RuntimeState(mode=self.engine.authority.mode.value)
        self.metrics = RuntimeMetrics()
        self.active = True

    def set_mode(self, mode: AuthorityMode, *, policy_enabled: bool | None = None) -> None:
        self.engine.set_mode(mode, policy_enabled=policy_enabled)
        self.session.authority = self.engine.authority
        self.state.mode = mode.value

    def stop(self) -> None:
        self.active = False

    def cycle(
        self,
        *,
        operator_intent: ActionIntent | None = None,
        observation: Observation | None = None,
    ) -> CycleResult:
        if not self.active:
            raise RuntimeError("runtime is stopped")
        obs = self.platform.observe() if observation is None else observation
        prior_focus = None if self.session.focus is None else self.session.focus.to_dict()
        prior_experience = 0 if self.session.self_model is None else len(self.session.self_model.experience.records)
        step: EngineStep = self.engine.step(obs, self.session, operator_intent=operator_intent)
        self.session = step.state
        result = self.platform.execute(step.action_intent)
        self._commit_cycle(step, result, prior_focus, prior_experience)
        return CycleResult(
            runtime=RuntimeState.from_dict(self.state.to_dict()),
            session=self.session,
            action_intent=step.action_intent,
            action_result=result,
            telemetry=step.telemetry,
            metrics=RuntimeMetrics.from_dict(self.metrics.to_dict()),
        )

    def run(self, n_cycles: int, *, operator_intent: ActionIntent | None = None) -> list[CycleResult]:
        out: list[CycleResult] = []
        for _ in range(n_cycles):
            if not self.active:
                break
            out.append(self.cycle(operator_intent=operator_intent))
        return out

    def save_checkpoint(self, path: str | Path) -> Path:
        path = Path(path)
        extras = {
            "identity": self.engine.identity_bundle(self.session),
            "runtime_state": self.state.to_dict(),
            "metrics": self.metrics.to_dict(),
            "session": {
                "cycle_id": self.session.cycle_id,
                "episode_position": float(self.session.episode_position),
                "last_intent": None if self.session.last_intent is None else self.session.last_intent.to_dict(),
                "world_meta": {"self_index": self.session.world.self_index, "provenance": self.session.world.provenance},
            },
            "memory_cursor": {
                "write_index": int(self.engine.system.memory.episodic.write_index),
                "reads": int(self.engine.system.memory.episodic.reads),
                "writes": int(self.engine.system.memory.episodic.writes),
            },
        }
        tensors = {
            "world": dump_world(self.session.world),
            "last_predicted": None if self.session.last_predicted is None else dump_world(self.session.last_predicted),
            "plant": self.platform.dump(),
        }
        saved = save_mina(path, self.engine.system, extras=extras, tensors=tensors)
        self.state.checkpoint_reference = saved.name
        return saved

    def restore_checkpoint(self, path: str | Path) -> RuntimeState:
        manifest, payload = load_mina(path, self.engine.system, return_payload=True)
        train = manifest.get("train") or {}
        bundle = train.get("identity") or {}
        tensors = payload.get("runtime") or {}
        world = load_world(tensors.get("world"), device=self.engine.device)
        if world is None:
            raise ValueError("runtime checkpoint missing WorldState")
        session_meta = train.get("session") or {}
        self.session = SessionState(
            cycle_id=int(session_meta.get("cycle_id", 0)),
            episode_position=float(session_meta.get("episode_position", 0.0)),
            world=world,
            last_intent=ActionIntent.from_dict(session_meta.get("last_intent")),
            last_predicted=load_world(tensors.get("last_predicted"), device=self.engine.device),
        )
        if bundle.get("self_model"):
            from minakanushi.identity.self_model import SelfModel

            self.session.self_model = SelfModel.from_dict(bundle["self_model"])
        if bundle.get("authority"):
            from minakanushi.identity.authority import AuthorityModel

            self.session.authority = AuthorityModel.from_dict(bundle["authority"])
            self.engine.authority = self.session.authority
        if bundle.get("persona"):
            from minakanushi.identity.persona import PersonaModel

            self.session.persona = PersonaModel.from_dict(bundle["persona"])
            self.engine.persona = self.session.persona
        if bundle.get("focus"):
            from minakanushi.focus.engine import FocusState

            self.session.focus = FocusState.from_dict(bundle["focus"])
        cursor = train.get("memory_cursor") or {}
        epi = self.engine.system.memory.episodic
        epi.write_index = int(cursor.get("write_index", epi.write_index))
        epi.reads = int(cursor.get("reads", epi.reads))
        epi.writes = int(cursor.get("writes", epi.writes))
        if tensors.get("plant") is not None:
            self.platform.restore(tensors["plant"])
        self.state = RuntimeState.from_dict(train.get("runtime_state"))
        self.metrics = RuntimeMetrics.from_dict(train.get("metrics"))
        self.metrics.checkpoint_restores += 1
        self.state.checkpoint_reference = Path(path).name
        self.active = True
        return self.state

    def _commit_cycle(
        self,
        step: EngineStep,
        result: ActionResult,
        prior_focus: dict | None,
        prior_experience: int,
    ) -> None:
        focus = {} if self.session.focus is None else self.session.focus.to_dict()
        intent = step.action_intent
        self.state.cycle_id = self.session.cycle_id
        self.state.runtime_time = float(result.timestamp)
        self.state.mode = (self.session.authority or self.engine.authority).mode.value
        self.state.current_situation_id = f"cycle-{self.session.cycle_id}"
        self.state.current_focus = focus
        self.state.active_prediction = belief_fingerprint(self.session.last_predicted or self.session.world)
        self.state.last_action_intent = intent.to_dict()
        self.state.last_action_result = result.to_dict()
        self.state.pending_experience = 1
        health = "ok"
        if self.session.self_model is not None:
            health = self.session.self_model.runtime.health_state
        self.state.health = health
        self.metrics.runtime_cycles += 1
        self.metrics.belief_updates += 1
        self.metrics.memory_writes += int(step.telemetry.memory_writes)
        if prior_focus != focus:
            self.metrics.focus_changes += 1
        self.metrics.prediction_updates += 1
        self.metrics.action_attempts += 1
        if str(intent.provenance).startswith("authority."):
            self.metrics.authority_blocks += 1
        exp_n = 0 if self.session.self_model is None else len(self.session.self_model.experience.records)
        self.metrics.experience_records = exp_n
