"""Gate G: no shortcut. Ablate channels / permute structure. Not a new head."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from minakanushi.perception.bridge import Observation
from simulations.synthetic_world.dataset import Episode


def _permute_visible(item: dict[str, Any]) -> dict[str, Any]:
    row = dict(item)
    xy = row.get("xy", (0.0, 0.0))
    vel = row.get("vel", (0.0, 0.0))
    row["xy"] = vel
    row["vel"] = xy
    return row


def mutate_observation(obs: Observation, mode: str) -> Observation:
    if mode == "full":
        return obs
    if mode == "drop_vision":
        return replace(obs, visible=(), noise_std=max(float(obs.noise_std), 1.0))
    if mode == "delay_telemetry":
        arrival = obs.arrival_time if obs.arrival_time is not None else obs.timestamp
        return replace(obs, arrival_time=float(arrival) + 0.5)
    if mode == "permute_structure":
        visible = tuple(_permute_visible(dict(item)) for item in obs.visible)
        return replace(
            obs,
            agent_xy=obs.agent_vel,
            agent_vel=obs.agent_xy,
            heading=float(obs.health),
            health=float(obs.heading),
            visible=visible,
        )
    raise ValueError(f"unknown shortcut mode {mode!r}")


def mutate_episode(episode: Episode, mode: str) -> Episode:
    return replace(
        episode,
        observations=[mutate_observation(obs, mode) for obs in episode.observations],
    )
