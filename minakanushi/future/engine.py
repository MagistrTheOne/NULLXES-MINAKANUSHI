"""Future Engine — branching futures that cannot mutate WorldState.

Invariant:
    OBSERVATION UPDATES REALITY.
    PREDICTION EXPLORES POSSIBILITY.
    POSSIBILITY MUST NOT BECOME REALITY WITHOUT NEW EVIDENCE.

P(future) and uncertainty(future) are independent heads.
Probabilities normalize within one strategy, never across unrelated strategies.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from minakanushi.architecture.config import ArchitectureConfig
from minakanushi.core.recurrent_state import clone_world
from minakanushi.future.belief import roll_belief
from minakanushi.future.trajectory import FutureTrajectory
from minakanushi.state.entity import AGENT_SLOT
from minakanushi.state.world import WorldState
from minakanushi.strategy.candidate import StrategyCandidate
from minakanushi.strategy.hold import HOLD_MODE, is_hold


class FutureEngine(nn.Module):
    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.latent_dim
        self.n_branches = config.future_branches
        self.branch_embed = nn.Parameter(torch.randn(self.n_branches, dim) * 0.02)
        self.residual = nn.Sequential(
            nn.Linear(dim * 2 + 4, dim),
            nn.SiLU(),
            nn.Linear(dim, 2),
        )
        # Independent heads: logit is not a function of uncertainty.
        self.branch_logit_head = nn.Linear(dim, 1)
        self.branch_unc_head = nn.Linear(dim, 1)

    def predict(
        self,
        world: WorldState,
        strategies: list[StrategyCandidate],
        max_horizon: int | None = None,
    ) -> list[FutureTrajectory]:
        original = (
            world.latent_state.data_ptr(),
            world.entity_xy.data_ptr(),
            world.entity_vel.data_ptr(),
            world.occupied.data_ptr(),
            world.uncertainty.data_ptr(),
        )
        snapshot = clone_world(world)
        horizon = max_horizon or self.config.prediction_horizons.medium
        dt = self.config.dt
        trajectories: list[FutureTrajectory] = []
        agent_xy = snapshot.entity_xy[:, AGENT_SLOT]
        occ = snapshot.occupied.to(snapshot.latent_state.dtype).unsqueeze(-1)
        pooled = (snapshot.latent_state * occ).sum(dim=1) / occ.sum(dim=1).clamp_min(1.0)

        for strategy in strategies:
            action_vec = self._action_vector(strategy, agent_xy)
            logits = []
            uncs = []
            stacked_list = []
            for k in range(self.n_branches):
                branch = self.branch_embed[k].view(1, 1, -1).expand_as(snapshot.latent_state)
                cond = torch.cat(
                    [
                        snapshot.latent_state,
                        branch,
                        action_vec.unsqueeze(1).expand(-1, snapshot.latent_state.shape[1], -1),
                    ],
                    dim=-1,
                )
                xy = snapshot.entity_xy.clone()
                vel = snapshot.entity_vel.clone()
                frames = []
                agent_vel = action_vec[:, :2]
                agent_mask = torch.zeros(vel.shape[0], vel.shape[1], 1, device=vel.device, dtype=torch.bool)
                agent_mask[:, AGENT_SLOT] = True
                for _ in range(horizon):
                    vel = torch.where(agent_mask, agent_vel.unsqueeze(1), vel)
                    residual = self.residual(cond)
                    xy = xy + vel * dt + residual * snapshot.occupied.unsqueeze(-1).to(xy.dtype)
                    frames.append(xy.clone())
                stacked = torch.stack(frames, dim=1)[0]
                stacked_list.append(stacked)
                context = pooled + self.branch_embed[k].unsqueeze(0)
                logits.append(self.branch_logit_head(context).squeeze(-1))
                uncs.append(torch.nn.functional.softplus(self.branch_unc_head(context).squeeze(-1)) + 1e-3)
            logit_t = torch.stack(logits, dim=-1)
            unc_t = torch.stack(uncs, dim=-1)
            probs = torch.softmax(logit_t, dim=-1)
            for k in range(self.n_branches):
                trajectories.append(
                    FutureTrajectory(
                        states_xy=stacked_list[k],
                        probability=probs[0, k],
                        uncertainty=unc_t[0, k],
                        causal_assumptions=(f"strategy={strategy.strategy_id}", f"branch={k}"),
                        terminal_xy=stacked_list[k][-1],
                        action_id=strategy.strategy_id,
                        strategy_id=strategy.strategy_id,
                        branch_id=k,
                        horizon_steps=horizon,
                        branch_logit=logit_t[0, k],
                    )
                )

        after = (
            world.latent_state.data_ptr(),
            world.entity_xy.data_ptr(),
            world.entity_vel.data_ptr(),
            world.occupied.data_ptr(),
            world.uncertainty.data_ptr(),
        )
        if after != original:
            raise RuntimeError("FutureEngine mutated WorldState tensors; possibility leaked into reality")
        return trajectories

    def predict_belief(
        self,
        world: WorldState,
        strategy: StrategyCandidate,
        steps: int = 1,
    ) -> WorldState:
        """Belief(t)+Action → Belief(t+dt). Does not mutate world."""
        original = (
            world.latent_state.data_ptr(),
            world.entity_xy.data_ptr(),
            world.entity_vel.data_ptr(),
            world.existence.data_ptr(),
        )
        predicted = roll_belief(world, strategy, steps=steps, dt=self.config.dt)
        after = (
            world.latent_state.data_ptr(),
            world.entity_xy.data_ptr(),
            world.entity_vel.data_ptr(),
            world.existence.data_ptr(),
        )
        if after != original:
            raise RuntimeError("predict_belief mutated current belief")
        return predicted

    def _action_vector(self, strategy: StrategyCandidate, agent_xy: Tensor) -> Tensor:
        zeros = torch.zeros(agent_xy.shape[0], 4, device=agent_xy.device, dtype=agent_xy.dtype)
        if is_hold(strategy.objective):
            zeros[:, 3] = HOLD_MODE[strategy.objective]
            return zeros
        target = torch.tensor(strategy.target_xy, device=agent_xy.device, dtype=agent_xy.dtype).unsqueeze(0)
        delta = target - agent_xy
        norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1e-6)
        directed = (delta / norm)
        zeros[:, :2] = directed
        zeros[:, 2] = 1.0
        return zeros


def group_by_strategy(trajectories: list[FutureTrajectory]) -> dict[str, list[FutureTrajectory]]:
    grouped: dict[str, list[FutureTrajectory]] = {}
    for traj in trajectories:
        grouped.setdefault(traj.strategy_id, []).append(traj)
    return grouped
