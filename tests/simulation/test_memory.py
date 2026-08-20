"""Memory retrieval must change later world latents (not decorative storage)."""

from __future__ import annotations

import torch

from helpers import build_engine
from simulations.synthetic_world.world import SyntheticWorld


def test_memory_changes_later_inference() -> None:
    engine = build_engine()
    world = SyntheticWorld(engine.config.simulation, seed=9)
    world.movers[0].xy[:] = world.agent.xy + 0.4
    state = engine.initialize()
    obs = world.observe()
    result = engine.step(obs, state)
    live = result.state.world.latent_state
    zeros = torch.zeros_like(live)
    hinted = engine.memory.hints(result.state.world, live_writes=live)
    blank = engine.memory.hints(result.state.world, live_writes=zeros)
    delta = float((hinted - blank).abs().sum())
    assert delta > 0.0
