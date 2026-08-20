"""ExperienceEngine — memory as lived prediction error, not tensor RAG.

A cycle writes:
  prediction (prior kinematics coasted)
  reality (new evidence)
  error
  correction
  lesson

The lesson may inflate unobserved velocity/position std later.
It never 50/50 with live evidence.
"""

from __future__ import annotations

import torch
from torch import Tensor

from minakanushi.identity.experience import (
    LESSON_CONSISTENT,
    LESSON_POSITION,
    LESSON_REVISION,
    LESSON_VELOCITY,
    ExperienceLog,
    ExperienceRecord,
)
from minakanushi.state.entity import AGENT_SLOT
from minakanushi.state.world import WorldState

VEL_ERROR_THRESHOLD = 0.25
XY_ERROR_THRESHOLD = 0.25
VEL_LESSON_BOOST = 0.35
XY_LESSON_BOOST = 0.15


def classify_lesson(*, error_xy: float, error_vel: float, correction_required: bool, correction_reason: str) -> str:
    if correction_required and correction_reason == "hypothesis_revision":
        return LESSON_REVISION
    if error_vel >= VEL_ERROR_THRESHOLD:
        return LESSON_VELOCITY
    if error_xy >= XY_ERROR_THRESHOLD:
        return LESSON_POSITION
    if correction_required:
        return LESSON_REVISION
    return LESSON_CONSISTENT


class ExperienceEngine:
    def record_cycle(
        self,
        previous: WorldState,
        updated: WorldState,
        dt: float,
        action: str,
        event_time: float,
    ) -> list[ExperienceRecord]:
        """Compare coasted prior belief to new evidence. Skip agent and first detects."""
        records: list[ExperienceRecord] = []
        corrections = {int(ev.entity_id): ev for ev in updated.corrections}
        n_slots = previous.occupied.shape[1]
        for s in range(n_slots):
            if s == AGENT_SLOT:
                continue
            if not bool(previous.occupied[0, s]):
                continue
            eid = int(previous.entity_id[0, s].item())
            if eid == 0:
                continue
            hit = (updated.entity_id[0] == eid) & updated.occupied[0]
            if not bool(hit.any()):
                continue
            slot = int(hit.nonzero(as_tuple=False)[0].item())
            if float(updated.age_unobserved[0, slot].item()) != 0.0:
                continue
            prev_xy = previous.entity_xy[0, s]
            prev_vel = previous.entity_vel[0, s]
            pred_xy = prev_xy + prev_vel * dt
            obs_xy = updated.entity_xy[0, slot]
            obs_vel = updated.entity_vel[0, slot]
            error_xy = float(torch.linalg.vector_norm(pred_xy - obs_xy).item())
            error_vel = float(torch.linalg.vector_norm(prev_vel - obs_vel).item())
            event = corrections.get(eid)
            lesson = classify_lesson(
                error_xy=error_xy,
                error_vel=error_vel,
                correction_required=event is not None,
                correction_reason="" if event is None else event.correction_reason,
            )
            records.append(
                ExperienceRecord(
                    event_time=event_time,
                    situation=f"entity={eid}",
                    action=action,
                    entity_id=eid,
                    predicted_xy=(float(pred_xy[0]), float(pred_xy[1])),
                    predicted_vel=(float(prev_vel[0]), float(prev_vel[1])),
                    observed_xy=(float(obs_xy[0]), float(obs_xy[1])),
                    observed_vel=(float(obs_vel[0]), float(obs_vel[1])),
                    error_xy=error_xy,
                    error_vel=error_vel,
                    correction_required=event is not None,
                    correction_reason="" if event is None else event.correction_reason,
                    lesson=lesson,
                )
            )
        return records

    def std_boost(self, world: WorldState, log: ExperienceLog) -> tuple[Tensor, Tensor]:
        """Return (xy_boost, vel_boost) [B, N, 2] for unobserved occupied slots."""
        xy_boost = torch.zeros_like(world.xy_std)
        vel_boost = torch.zeros_like(world.vel_std)
        for b in range(world.entity_id.shape[0]):
            for s in range(world.entity_id.shape[1]):
                if not bool(world.occupied[b, s]):
                    continue
                eid = int(world.entity_id[b, s].item())
                rec = log.latest_for(eid)
                if rec is None:
                    continue
                if rec.lesson == LESSON_VELOCITY:
                    vel_boost[b, s] = VEL_LESSON_BOOST
                elif rec.lesson == LESSON_POSITION:
                    xy_boost[b, s] = XY_LESSON_BOOST
                elif rec.lesson == LESSON_REVISION:
                    xy_boost[b, s] = XY_LESSON_BOOST
                    vel_boost[b, s] = VEL_LESSON_BOOST * 0.5
        return xy_boost.clamp_min(0.0), vel_boost.clamp_min(0.0)
