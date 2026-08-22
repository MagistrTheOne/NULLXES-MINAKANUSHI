"""Architectural metrics. Loss is not a substitute for these."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class MetricBundle:
    world_state_position_error: float
    world_state_velocity_error: float
    future_ADE: float
    future_FDE: float
    entity_persistence_accuracy: float
    reacquisition_accuracy: float
    uncertainty_calibration_error: float
    branch_diversity: float
    branch_coverage: float
    memory_effect_delta: float
    constraint_violation_count: int
    closed_loop_success_rate: float
    belief_revision_accuracy: float = 0.0
    correction_latency: float = -1.0
    false_persistence_steps: float = 0.0
    evidence_dominance: float = 0.0
    revision_detected: float = 0.0
    revision_direction_accuracy: float = 0.0
    revision_magnitude_error: float = 0.0
    revision_latency: float = -1.0
    false_revision_rate: float = 0.0
    revision_accuracy: float = 0.0
    memory_future_delta: float = 0.0
    future_diversity: float = 0.0
    counterfactual_quality: float = 0.0
    memory_ade_on: float = 0.0
    memory_ade_off: float = 0.0
    memory_fde_on: float = 0.0
    memory_fde_off: float = 0.0
    memory_helps_future: float = 0.0


def masked_mse(pred: Tensor, true: Tensor, mask: Tensor) -> Tensor:
    w = mask.to(pred.dtype).unsqueeze(-1)
    return (pred - true).pow(2).mul(w).sum() / w.sum().clamp_min(1.0)


def displacement_error(pred_traj: Tensor, true_traj: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """pred_traj [B,H,N,2], mask [B,N] -> ADE, FDE scalars."""
    w = mask.to(pred_traj.dtype).unsqueeze(1).unsqueeze(-1)
    err = torch.linalg.vector_norm(pred_traj - true_traj, dim=-1)
    ade = (err * w.squeeze(-1)).sum() / w.sum().clamp_min(1.0)
    fde = (err[:, -1] * mask.to(err.dtype)).sum() / mask.to(err.dtype).sum().clamp_min(1.0)
    return ade, fde


def branch_diversity(branch_xy: Tensor) -> Tensor:
    """branch_xy [K, H, N, 2] pairwise terminal distance mean."""
    terminals = branch_xy[:, -1]
    k = terminals.shape[0]
    if k < 2:
        return terminals.new_zeros(())
    total = terminals.new_zeros(())
    count = 0
    for i in range(k):
        for j in range(i + 1, k):
            total = total + torch.linalg.vector_norm(terminals[i] - terminals[j], dim=-1).mean()
            count += 1
    return total / max(count, 1)


def memory_effect_delta(with_memory: Tensor, without_memory: Tensor, mask: Tensor) -> Tensor:
    """L2 change in occupied latents when retrieval is enabled vs zeros."""
    w = mask.to(with_memory.dtype).unsqueeze(-1)
    return (with_memory - without_memory).pow(2).mul(w).sum() / w.sum().clamp_min(1.0)


def belief_revision_accuracy(old_xy: Tensor, new_xy: Tensor, evidence_xy: Tensor) -> Tensor:
    """1 if the update moved toward evidence, 0 if it stayed or moved away."""
    before = torch.linalg.vector_norm(old_xy - evidence_xy, dim=-1)
    after = torch.linalg.vector_norm(new_xy - evidence_xy, dim=-1)
    return (after < before).to(old_xy.dtype)


def correction_latency(wrong_at: int, evidence_at: int, corrected_at: int) -> int:
    """Steps from correct evidence to completed revision. -1 if never corrected."""
    if corrected_at < 0 or evidence_at < 0:
        return -1
    return max(0, corrected_at - evidence_at)


def false_persistence_steps(occupied_after_gone: list[bool]) -> int:
    """How many steps a vanished entity stays occupied. Memory without decay hallucinates."""
    n = 0
    for flag in occupied_after_gone:
        if not flag:
            break
        n += 1
    return n


def evidence_dominance(result: float, belief: float, evidence: float) -> float:
    """1 if result is closer to evidence than to the midpoint (not blind average)."""
    mid = 0.5 * (belief + evidence)
    return 1.0 if abs(result - evidence) < abs(result - mid) else 0.0


def count_hard_violations(candidate, trajectories, simulation) -> int:
    """Count HARD rule failures on (candidate, each predicted branch)."""
    from minakanushi.constraints.rule import ConstraintClass, RULE_REGISTRY

    rules = tuple(RULE_REGISTRY[name]() for name in simulation.hard_constraints if name in RULE_REGISTRY)
    checks = list(trajectories) if trajectories else [None]
    n = 0
    for traj in checks:
        for rule in rules:
            if rule.cls != ConstraintClass.HARD:
                continue
            ok, _ = rule.evaluate(candidate, traj, simulation)
            if not ok:
                n += 1
    return n


def closed_loop_success(simulation, intent, agent_xy, agent_vel, timestamp: float) -> float:
    """One observe-after-act cycle. 1.0 only if time advances, agent stays in arena, and the executed command passes HARD rules."""
    import numpy as np

    from minakanushi.strategy.candidate import StrategyCandidate
    from simulations.synthetic_world.world import SyntheticWorld

    world = SyntheticWorld(simulation, seed=0)
    world.agent.xy = np.array(agent_xy, dtype=np.float64)
    world.agent.vel = np.array(agent_vel, dtype=np.float64)
    world.t = float(timestamp)
    t0 = world.t
    world.step(intent)
    nxt = world.observe()
    if nxt.timestamp <= t0:
        return 0.0
    x, y = float(world.agent.xy[0]), float(world.agent.xy[1])
    x0, x1, y0, y1 = simulation.arena
    if x < x0 or x > x1 or y < y0 or y > y1:
        return 0.0
    executed = StrategyCandidate(intent.strategy_id, intent.objective, intent.target_state, 0.0, 0.0)
    if count_hard_violations(executed, (), simulation) > 0:
        return 0.0
    return 1.0


def policy_firewall_metrics(kernel, policy, candidates, trajectories, simulation, goal_xy, now, agent_xy, agent_vel) -> tuple[int, float]:
    """Measure selected-intent HARD violations and a real closed-loop step. Not a constant."""
    from minakanushi.strategy.candidate import StrategyCandidate

    allowed, rejected, _ = kernel.filter(list(candidates), trajectories)
    intent = policy.select(allowed, trajectories, goal_xy, now)
    rejected_ids = {c.strategy_id for c in rejected}
    violations = 1 if intent.strategy_id in rejected_ids else 0
    selected = None
    for item in allowed:
        if item.strategy_id == intent.strategy_id:
            selected = item.candidate
            break
    if selected is None:
        selected = StrategyCandidate(intent.strategy_id, intent.objective, intent.target_state, 0.0, 0.0)
    violations += count_hard_violations(selected, trajectories.get(intent.strategy_id, []), simulation)
    success = closed_loop_success(simulation, intent, agent_xy, agent_vel, now)
    return violations, success


def action_influence_score(future_a: Tensor, future_b: Tensor, mask: Tensor) -> Tensor:
    """Does the action change the forecast? future_* [N, 2], mask [N]."""
    w = mask.to(future_a.dtype).unsqueeze(-1)
    return ((future_a - future_b).pow(2).mul(w).sum() / w.sum().clamp_min(1.0)).sqrt()


def counterfactual_separation_score(future_a: Tensor, future_b: Tensor) -> Tensor:
    return torch.linalg.vector_norm(future_a - future_b, dim=-1).mean()


def causal_consistency_score(agent_delta: Tensor, other_delta: Tensor) -> Tensor:
    """Action should move the agent more than an unrelated object. Higher is better."""
    a = torch.linalg.vector_norm(agent_delta, dim=-1).mean()
    o = torch.linalg.vector_norm(other_delta, dim=-1).mean()
    return a - o


def prediction_calibration_score(confidence: Tensor, error: Tensor) -> Tensor:
    """confidence in [0,1], error >= 0. Well-calibrated: high conf ↔ low error."""
    hit = (error < 0.5).to(confidence.dtype)
    return (confidence - hit).abs().mean()


def assemble_bundle(
    pred_xy: Tensor,
    true_xy: Tensor,
    pred_vel: Tensor,
    true_vel: Tensor,
    occupied: Tensor,
    pred_future: Tensor,
    true_future: Tensor,
    persist_hits: float,
    reacquire_hits: float,
    uncertainty: Tensor,
    position_error: Tensor,
    branch_xy: Tensor,
    memory_delta: float,
    constraint_violations: int,
    closed_loop_success: float,
    coverage: float,
    before_xy: Tensor | None = None,
    after_xy: Tensor | None = None,
    evidence_xy: Tensor | None = None,
    should_revise: Tensor | None = None,
    has_evidence: Tensor | None = None,
    occupied_before: Tensor | None = None,
    entity_id: Tensor | None = None,
    memory_future_delta: float = 0.0,
    future_diversity: float = 0.0,
    counterfactual_quality: float = 0.0,
    memory_ade_on: float = 0.0,
    memory_ade_off: float = 0.0,
    memory_fde_on: float = 0.0,
    memory_fde_off: float = 0.0,
    memory_helps_future: float = 0.0,
) -> MetricBundle:
    pos = masked_mse(pred_xy, true_xy, occupied)
    vel = masked_mse(pred_vel, true_vel, occupied)
    ade, fde = displacement_error(pred_future, true_future, occupied)
    u_mean = uncertainty.mean(dim=-1)
    err = torch.linalg.vector_norm(position_error, dim=-1)
    cal = (u_mean - err).abs()
    cal = (cal * occupied.to(cal.dtype)).sum() / occupied.to(cal.dtype).sum().clamp_min(1.0)
    rev = {
        "revision_detected": 0.0,
        "revision_direction_accuracy": 0.0,
        "revision_magnitude_error": 0.0,
        "revision_latency": -1.0,
        "false_revision_rate": 0.0,
        "belief_revision_accuracy": 0.0,
    }
    if (
        before_xy is not None
        and after_xy is not None
        and evidence_xy is not None
        and should_revise is not None
        and has_evidence is not None
        and occupied_before is not None
        and entity_id is not None
    ):
        from minakanushi.training.revision import revision_metrics

        rev = revision_metrics(
            before_xy=before_xy,
            after_xy=after_xy,
            evidence_xy=evidence_xy,
            should_revise=should_revise,
            has_evidence=has_evidence,
            occupied_before=occupied_before,
            entity_id=entity_id,
        )
    return MetricBundle(
        world_state_position_error=float(pos.detach()),
        world_state_velocity_error=float(vel.detach()),
        future_ADE=float(ade.detach()),
        future_FDE=float(fde.detach()),
        entity_persistence_accuracy=float(persist_hits),
        reacquisition_accuracy=float(reacquire_hits),
        uncertainty_calibration_error=float(cal.detach()),
        branch_diversity=float(branch_diversity(branch_xy).detach()),
        branch_coverage=float(coverage),
        memory_effect_delta=float(memory_delta),
        constraint_violation_count=int(constraint_violations),
        closed_loop_success_rate=float(closed_loop_success),
        belief_revision_accuracy=float(rev["belief_revision_accuracy"]),
        correction_latency=float(rev["revision_latency"]),
        revision_detected=float(rev["revision_detected"]),
        revision_direction_accuracy=float(rev["revision_direction_accuracy"]),
        revision_magnitude_error=float(rev["revision_magnitude_error"]),
        revision_latency=float(rev["revision_latency"]),
        false_revision_rate=float(rev["false_revision_rate"]),
        revision_accuracy=float(rev["belief_revision_accuracy"]),
        memory_future_delta=float(memory_future_delta),
        future_diversity=float(future_diversity),
        counterfactual_quality=float(counterfactual_quality),
        memory_ade_on=float(memory_ade_on),
        memory_ade_off=float(memory_ade_off),
        memory_fde_on=float(memory_fde_on),
        memory_fde_off=float(memory_fde_off),
        memory_helps_future=float(memory_helps_future),
    )

