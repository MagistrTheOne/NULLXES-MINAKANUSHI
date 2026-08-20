"""ActionOutcomeRecord — predicted transition vs actual after an act.

This is not a chat log and not a substitute for ExperienceRecord.
Experience is lived kinematics. ActionOutcome is causal: I did A, I expected B, I saw C.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from minakanushi.policy.intent import ActionIntent
from minakanushi.state.world import WorldState

SUCCESS_ERROR = 0.5
CORRECTION_ERROR = 0.25


def _slots(world: WorldState) -> list[dict]:
    rows = []
    occ = world.occupied[0]
    for slot in occ.nonzero(as_tuple=False).flatten().tolist():
        eid = int(world.entity_id[0, slot].item())
        rows.append(
            {
                "entity_id": eid,
                "xy": [float(world.entity_xy[0, slot, 0]), float(world.entity_xy[0, slot, 1])],
                "vel": [float(world.entity_vel[0, slot, 0]), float(world.entity_vel[0, slot, 1])],
                "existence": float(world.existence[0, slot]),
            }
        )
    return rows


def snapshot_belief(world: WorldState) -> dict:
    return {"timestamp": float(world.timestamp[0].item()), "slots": _slots(world)}


def transition_error(predicted: WorldState, actual: WorldState) -> tuple[float, float]:
    """Return (xy_error, vel_error) averaged over shared entity ids."""
    pred = {row["entity_id"]: row for row in _slots(predicted)}
    act = {row["entity_id"]: row for row in _slots(actual)}
    shared = [eid for eid in pred if eid in act]
    if not shared:
        return 1.0, 1.0
    xy = 0.0
    vel = 0.0
    for eid in shared:
        a = torch.tensor(pred[eid]["xy"])
        b = torch.tensor(act[eid]["xy"])
        va = torch.tensor(pred[eid]["vel"])
        vb = torch.tensor(act[eid]["vel"])
        xy += float(torch.linalg.vector_norm(a - b))
        vel += float(torch.linalg.vector_norm(va - vb))
    n = float(len(shared))
    return xy / n, vel / n


@dataclass(frozen=True)
class ActionOutcomeRecord:
    event_time: float
    action_intent: dict
    belief_before: dict
    predicted_transition: dict
    actual_transition: dict
    prediction_error: float
    success: bool
    correction: bool
    lesson: str

    def to_dict(self) -> dict:
        return {
            "event_time": self.event_time,
            "action_intent": dict(self.action_intent),
            "belief_before": dict(self.belief_before),
            "predicted_transition": dict(self.predicted_transition),
            "actual_transition": dict(self.actual_transition),
            "prediction_error": self.prediction_error,
            "success": self.success,
            "correction": self.correction,
            "lesson": self.lesson,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> ActionOutcomeRecord:
        return cls(
            event_time=float(raw["event_time"]),
            action_intent=dict(raw.get("action_intent", {})),
            belief_before=dict(raw.get("belief_before", {})),
            predicted_transition=dict(raw.get("predicted_transition", {})),
            actual_transition=dict(raw.get("actual_transition", {})),
            prediction_error=float(raw.get("prediction_error", 0.0)),
            success=bool(raw.get("success", True)),
            correction=bool(raw.get("correction", False)),
            lesson=str(raw.get("lesson", "consistent")),
        )


def record_outcome(
    *,
    intent: ActionIntent,
    belief_before: WorldState,
    predicted: WorldState,
    actual: WorldState,
    event_time: float,
) -> ActionOutcomeRecord:
    xy_err, vel_err = transition_error(predicted, actual)
    err = xy_err + vel_err
    if vel_err >= CORRECTION_ERROR:
        lesson = "visibility_or_velocity_model"
    elif xy_err >= CORRECTION_ERROR:
        lesson = "transition_mismatch"
    else:
        lesson = "consistent"
    return ActionOutcomeRecord(
        event_time=event_time,
        action_intent={
            "strategy_id": intent.strategy_id,
            "objective": intent.objective,
            "target_state": list(intent.target_state),
        },
        belief_before=snapshot_belief(belief_before),
        predicted_transition=snapshot_belief(predicted),
        actual_transition=snapshot_belief(actual),
        prediction_error=err,
        success=err < SUCCESS_ERROR,
        correction=err >= CORRECTION_ERROR,
        lesson=lesson,
    )


@dataclass
class ActionOutcomeLog:
    records: list[ActionOutcomeRecord] = field(default_factory=list)
    cap: int = 64

    def append(self, record: ActionOutcomeRecord) -> None:
        self.records.append(record)
        if len(self.records) > self.cap:
            self.records = self.records[-self.cap :]

    def to_dict(self) -> dict:
        return {"cap": self.cap, "records": [r.to_dict() for r in self.records]}

    @classmethod
    def from_dict(cls, raw: dict) -> ActionOutcomeLog:
        log = cls(cap=int(raw.get("cap", 64)))
        log.records = [ActionOutcomeRecord.from_dict(item) for item in raw.get("records", [])]
        return log
