"""S3: weighted value with overspeed penalty + CEM refinement."""

from __future__ import annotations

import torch

from minakanushi.future.engine import cem_refine
from minakanushi.state.constructor import StateConstructor, empty_world_state
from minakanushi.strategy.candidate import StrategyCandidate
from minakanushi.strategy.evaluator import evaluate_value
from helpers import cpu_config


def _world():
    cfg = cpu_config()
    world = empty_world_state(cfg.architecture, 1, device=torch.device("cpu"), dtype=torch.float32)
    return cfg, world


def test_overspeed_penalty_prefers_compliant() -> None:
    from minakanushi.future.trajectory import FutureTrajectory

    def traj_at(x: float):
        term = torch.zeros(16, 2)
        term[0] = torch.tensor([x, 0.0])
        return FutureTrajectory(
            states_xy=torch.zeros(2, 16, 2), probability=torch.tensor(1.0),
            uncertainty=torch.tensor(0.1), causal_assumptions=(), terminal_xy=term,
            action_id="m", strategy_id="m", branch_id=0, horizon_steps=2,
        )

    fast = StrategyCandidate("m", "MOVE_TO", (9.0, 0.0), 0.0, 0.0, parameters={"speed": 5.0})
    slow = StrategyCandidate("m", "MOVE_TO", (9.0, 0.0), 0.0, 0.0, parameters={"speed": 0.5})
    t = traj_at(9.0)
    assert evaluate_value(fast, t, (9.0, 0.0), max_speed=1.2) < evaluate_value(slow, t, (9.0, 0.0), max_speed=1.2)
    # Legacy default path unchanged (no max_speed -> no penalty term).
    assert evaluate_value(fast, t, (9.0, 0.0)) == evaluate_value(slow, t, (9.0, 0.0))


def test_cem_never_regresses_and_is_deterministic() -> None:
    cfg, world = _world()
    from minakanushi.future.engine import FutureEngine

    engine = FutureEngine(cfg.architecture)
    seed = StrategyCandidate("move_to_9", "MOVE_TO", (9.0, 9.0), 0.0, 0.2, parameters={"speed": 1.0})
    goal = (9.0, 9.0)
    best1, val1 = cem_refine(engine, world, seed, goal, max_speed=1.2, iters=2, samples=4, std=0.5, rng_seed=7)
    best2, val2 = cem_refine(engine, world, seed, goal, max_speed=1.2, iters=2, samples=4, std=0.5, rng_seed=7)
    assert (best1.target_xy, val1) == (best2.target_xy, val2)
    from minakanushi.future.engine import group_by_strategy

    seed_trajs = engine.predict(world, [seed], max_horizon=cfg.architecture.prediction_horizons.short, max_speed=1.2)
    seed_traj = max(seed_trajs, key=lambda t: float(t.probability.detach()))
    seed_val = evaluate_value(seed, seed_traj, goal, max_speed=1.2)
    # Seed clone competes in the batch, so best never regresses vs seed.
    assert val1 >= seed_val - 1e-6


def test_planner_config_defaults_off() -> None:
    cfg = cpu_config()
    assert cfg.architecture.planner.enabled is False
    assert cfg.architecture.planner.iters >= 1
    assert cfg.architecture.planner.samples >= 1
