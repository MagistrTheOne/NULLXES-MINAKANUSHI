"""Stage-0/2 trainer with live gradient paths for every named loss."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from minakanushi.architecture.config import MinakanushiConfig, load_config
from minakanushi.architecture.mina_unit import MinaUnitBatch, pack_units
from minakanushi.architecture.model import MinakanushiSystem
from minakanushi.state.constructor import StateConstructor, empty_world_state
from minakanushi.strategy.candidate import StrategyCandidate
from minakanushi.training.checkpoint import save_mina
from minakanushi.training.objectives import compute_objectives
from minakanushi.utils.seed import seed_everything
from minakanushi.utils.tensors import assert_finite, resolve_device, resolve_dtype
from simulations.synthetic_world.dataset import generate_episode


@dataclass
class TrainLog:
    step: int
    loss: float
    terms: dict[str, float]
    grad_norm: float
    traj_error: float


def _align(pred_ids: Tensor, pred_occ: Tensor, true_ids: Tensor, true_xy: Tensor, true_vel: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Match predicted slots to GT by entity_id. Discrete matching, no gradient."""
    xy = torch.zeros_like(pred_ids, dtype=true_xy.dtype).unsqueeze(-1).repeat(1, 1, 2)
    vel = torch.zeros_like(xy)
    occ = torch.zeros_like(pred_occ)
    xy = xy.clone()
    vel = vel.clone()
    for b in range(pred_ids.shape[0]):
        for s in range(pred_ids.shape[1]):
            if not bool(pred_occ[b, s]):
                continue
            eid = int(pred_ids[b, s].item())
            hits = (true_ids[b] == eid).nonzero(as_tuple=False)
            if hits.numel() == 0:
                continue
            j = int(hits[0].item())
            xy[b, s] = true_xy[b, j]
            vel[b, s] = true_vel[b, j]
            occ[b, s] = True
    return xy, vel, occ


def _truth_tensors(truth, max_entities: int, device, dtype) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    n = min(len(truth.entity_id), max_entities)
    ids = torch.zeros(1, max_entities, dtype=torch.long, device=device)
    xy = torch.zeros(1, max_entities, 2, device=device, dtype=dtype)
    vel = torch.zeros(1, max_entities, 2, device=device, dtype=dtype)
    occ = torch.zeros(1, max_entities, dtype=torch.bool, device=device)
    for i in range(n):
        ids[0, i] = int(truth.entity_id[i])
        xy[0, i, 0] = float(truth.xy[i, 0])
        xy[0, i, 1] = float(truth.xy[i, 1])
        vel[0, i, 0] = float(truth.vel[i, 0])
        vel[0, i, 1] = float(truth.vel[i, 1])
        occ[0, i] = True
    return ids, xy, vel, occ


class Trainer:
    def __init__(self, config: MinakanushiConfig, root: Path) -> None:
        if config.training is None:
            raise ValueError("Trainer requires training config")
        self.config = config
        self.root = root
        seed_everything(config.training.seed)
        self.device = resolve_device(config.training.device)
        self.dtype = resolve_dtype(config.training.precision)
        self.system = MinakanushiSystem(config.architecture).to(self.device)
        self.constructor = StateConstructor(config.architecture)
        self.opt = torch.optim.AdamW(
            self.system.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

    def _encode(self, obs, episode_pos: float) -> MinaUnitBatch:
        units = self.system.perception.encode(obs, device=self.device, dtype=self.dtype)
        now = obs.arrival_time if obs.arrival_time is not None else obs.timestamp
        return pack_units(
            units,
            batch_index=0,
            max_units=self.config.architecture.max_observations,
            latent_dim=self.config.architecture.latent_dim,
            episode_position=episode_pos,
            now=now,
            device=self.device,
            dtype=self.dtype,
        )

    def _core_step(self, packed, world, live_writes):
        hints = self.system.memory.hints(world, live_writes=live_writes)
        pos = self.system.position_units(packed)
        fused = packed.semantic_embedding + pos.embedding
        constructed = self.constructor.apply(packed, world, fused, memory_hints=hints)
        _, core = self.system.observe_to_core(packed, constructed, hints)
        return core

    def step_once(self, step: int) -> TrainLog:
        train = self.config.training
        arch = self.config.architecture
        ep_idx = (step - 1) % max(train.n_overfit_episodes, 1)
        episode = generate_episode(
            self.config.simulation,
            seed=train.seed,
            episode_index=ep_idx,
            length=train.sequence_length,
            horizon=arch.prediction_horizons.short,
        )
        idx = min(3, len(episode.observations) - 3)
        obs = episode.observations[idx]
        obs_n = episode.observations[idx + 1]
        truth = episode.truth[idx]
        truth_n = episode.truth[idx + 1]

        world = empty_world_state(arch, 1, device=self.device, dtype=self.dtype)
        world.entity_xy[0, 0] = torch.tensor(self.config.simulation.agent_start, device=self.device, dtype=self.dtype)

        packed = self._encode(obs, float(idx))
        core = self._core_step(packed, world, live_writes=None)
        writes = core.memory_write_candidates
        pred = core.world_state

        packed_n = self._encode(obs_n, float(idx + 1))
        core_n = self._core_step(packed_n, pred, live_writes=writes)
        pred_n = core_n.world_state

        max_e = arch.world_slots
        true_ids, true_xy, true_vel, _ = _truth_tensors(truth, max_e, self.device, self.dtype)
        aligned_xy, aligned_vel, aligned_occ = _align(pred.entity_id, pred.occupied, true_ids, true_xy, true_vel)
        true_ids_n, true_next, true_next_vel, _ = _truth_tensors(truth_n, max_e, self.device, self.dtype)
        aligned_next, aligned_next_vel, aligned_occ_n = _align(
            pred_n.entity_id, pred_n.occupied, true_ids_n, true_next, true_next_vel
        )

        occluded = set(truth_n.occluded_ids) | (set(truth.visible_ids) - set(truth_n.visible_ids))
        mem_mask = torch.zeros_like(aligned_occ)
        for s in range(arch.world_slots):
            if bool(pred_n.occupied[0, s]) and int(pred_n.entity_id[0, s].item()) in occluded:
                mem_mask[0, s] = True
        if not bool(mem_mask.any()):
            mem_mask = aligned_occ_n

        cand = StrategyCandidate(truth.action.lower(), truth.action, truth.action_target, 0.0, 0.0)
        alt = StrategyCandidate("wait", "WAIT", tuple(float(x) for x in episode.observations[idx].agent_xy), 0.0, 0.0)
        trajs = self.system.future.predict(pred, [cand, alt], max_horizon=arch.prediction_horizons.short)
        primary = [t for t in trajs if t.strategy_id == cand.strategy_id]
        other = [t for t in trajs if t.strategy_id == alt.strategy_id]
        pred_future = primary[0].states_xy.unsqueeze(0)
        alt_future = other[0].states_xy.unsqueeze(0)
        intra_b = primary[1].states_xy.unsqueeze(0) if len(primary) > 1 else alt_future
        h = pred_future.shape[1]
        true_future = aligned_next.unsqueeze(1).expand(-1, h, -1, -1).contiguous()
        for k in range(h):
            for s in range(arch.world_slots):
                if not bool(pred.occupied[0, s]):
                    continue
                eid = int(pred.entity_id[0, s].item())
                if eid in truth.future_xy:
                    true_future[0, k, s, 0] = float(truth.future_xy[eid][k, 0])
                    true_future[0, k, s, 1] = float(truth.future_xy[eid][k, 1])

        breakdown = compute_objectives(
            pred_xy=pred.entity_xy,
            true_xy=aligned_xy,
            occupied=aligned_occ,
            pred_next_xy=pred_n.entity_xy,
            true_next_xy=aligned_next,
            pred_future_xy=pred_future,
            true_future_xy=true_future,
            uncertainty=pred.uncertainty,
            memory_xy=pred_n.entity_xy,
            memory_true_xy=aligned_next,
            memory_mask=mem_mask,
            causal_pred=pred.entity_vel,
            causal_true=aligned_vel,
            alt_future_xy=alt_future,
            intra_branch_xy=intra_b,
            latent=pred.latent_state,
            training=train,
        )
        assert_finite("loss.total", breakdown.total)
        self.opt.zero_grad(set_to_none=True)
        breakdown.total.backward()
        grad_norm = float(clip_grad_norm_(self.system.parameters(), train.grad_clip))
        self.opt.step()
        traj_error = float(((pred_future[:, -1] - true_future[:, -1]).pow(2) * aligned_occ.unsqueeze(-1)).sum().item())
        return TrainLog(
            step=step,
            loss=float(breakdown.total.item()),
            terms={k: float(v.item()) for k, v in breakdown.terms.items()},
            grad_norm=grad_norm,
            traj_error=traj_error,
        )

    def fit(self, out_dir: Path) -> list[TrainLog]:
        train = self.config.training
        logs: list[TrainLog] = []
        out_dir.mkdir(parents=True, exist_ok=True)
        for step in range(1, train.steps + 1):
            log = self.step_once(step)
            logs.append(log)
            if step % train.log_every == 0 or step == 1:
                print(
                    f"step={step} loss={log.loss:.4f} traj_err={log.traj_error:.4f} grad={log.grad_norm:.4f} terms={log.terms}",
                    flush=True,
                )
            if step % train.checkpoint_every == 0 or step == train.steps:
                save_mina(
                    out_dir / f"minakanushi_stage{train.stage}_step{step}.mina",
                    self.system,
                    optimizer=self.opt,
                    extras={
                        "stage": train.stage,
                        "step": step,
                        "seed": train.seed,
                        "loss": log.loss,
                        "traj_error": log.traj_error,
                        "metrics": {"traj_error": log.traj_error, "loss": log.loss},
                    },
                )
        return logs


def trainer_from_files(root: Path, training_yaml: Path) -> Trainer:
    from minakanushi.architecture.config import load_training

    training = load_training(training_yaml)
    config = load_config(
        root / training.architecture,
        training_path=training_yaml,
        runtime_path=root / "configs" / "runtime" / "cpu.yaml",
        simulation_path=root / training.simulation,
    )
    return Trainer(config, root)
