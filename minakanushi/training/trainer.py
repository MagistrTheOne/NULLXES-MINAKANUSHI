"""Stage-0/2 trainer with live gradient paths for every named loss."""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch import Tensor

from minakanushi.architecture.config import MinakanushiConfig, load_config
from minakanushi.architecture.freeze import assert_may_construct, is_6_8b_profile
from minakanushi.architecture.mina_unit import MinaUnitBatch, pack_units
from minakanushi.architecture.model import MinakanushiSystem
from minakanushi.constraints.kernel import MinakanushiConstraintKernel
from minakanushi.future.engine import group_by_strategy
from minakanushi.policy.action_policy import ActionPolicy
from minakanushi.state.constructor import StateConstructor, empty_world_state
from minakanushi.strategy.candidate import StrategyCandidate
from minakanushi.strategy.hold import HOLD_MODE
from minakanushi.training.checkpoint import save_mina
from minakanushi.training.episode_dataset import JsonEpisodeDataset
from minakanushi.training.metrics import (
    assemble_bundle,
    branch_diversity,
    counterfactual_separation_score,
    displacement_error,
    masked_mse,
    memory_effect_delta,
    policy_firewall_metrics,
)
from minakanushi.training.phase_sampler import PhaseCurriculumSampler, mode_for_job_step
from minakanushi.training.objectives import compute_objectives
from minakanushi.training.parallel import clip_grad_norm_mixed, collect_full_checkpoint, dist_barrier, is_rank0, training_device
from minakanushi.training.resume import WarmupScheduler, apply_resume
from minakanushi.utils.seed import capture_rng, seed_everything
from minakanushi.utils.tensors import assert_finite, resolve_dtype
from minakanushi.training.revision import evidence_for_slots, should_revise_mask
from minakanushi.identity.initialize import canonical_identity_payload
from simulations.synthetic_world.dataset import TRAIN_CURRICULUM, generate_episode, training_frame


def _move_target(simulation, agent_xy: tuple[float, float], labeled_target) -> tuple[float, float]:
    if getattr(simulation, "targets", ()):
        xy = simulation.targets[0]["xy"]
        tgt = (float(xy[0]), float(xy[1]))
        if tgt != agent_xy:
            return tgt
    if labeled_target is not None:
        tgt = (float(labeled_target[0]), float(labeled_target[1]))
        if tgt != agent_xy:
            return tgt
    return (float(agent_xy[0] + 1.0), float(agent_xy[1]))


def counterfactual_candidate(truth, agent_xy: tuple[float, float], simulation) -> StrategyCandidate:
    """Second strategy must differ from the labeled action. WAIT is not MOVE_TO."""
    action = str(truth.action)
    if action in HOLD_MODE:
        target = _move_target(simulation, agent_xy, truth.action_target)
        return StrategyCandidate("move_to", "MOVE_TO", target, 0.0, 0.0)
    return StrategyCandidate("wait", "WAIT", agent_xy, 0.0, 0.0)


@dataclass
class TrainLog:
    step: int
    loss: float
    terms: dict[str, float]
    grad_norm: float
    traj_error: float
    metrics: dict[str, float] | None = None


@dataclass
class UnrollPacket:
    packed: MinaUnitBatch
    packed_n: MinaUnitBatch
    pos: object
    hints: Tensor
    writes: Tensor
    pred: object
    pred_n: object
    core: object
    core_n: object
    trajs: list
    breakdown: object
    aligned_xy: Tensor
    aligned_vel: Tensor
    aligned_occ: Tensor
    aligned_next: Tensor
    mem_mask: Tensor
    pred_future: Tensor
    true_future: Tensor
    alt_future: Tensor
    intra_b: Tensor
    episode_index: int
    frame_index: int
    scenario: str
    candidates: list
    obs_timestamp: float
    obs_agent_xy: tuple[float, float]
    obs_agent_vel: tuple[float, float]
    before_xy: Tensor
    evidence_xy: Tensor
    should_revise: Tensor
    has_evidence: Tensor
    occupied_before: Tensor
    n_constructor_corrections: int


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
    def __init__(self, config: MinakanushiConfig, root: Path, *, eval_only: bool = False) -> None:
        if config.training is None:
            raise ValueError("Trainer requires training config")
        train = config.training
        gpu_name = ""
        if str(train.device).startswith("cuda") and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
        assert_may_construct(config.architecture, device=train.device, gpu_name=gpu_name)
        if is_6_8b_profile(config.architecture):
            from minakanushi.training.parallel import plan_from_training

            plan_from_training(config.architecture, train)
            if train.parallelism == "fsdp2_zero3":
                import torch.distributed as dist

                if not dist.is_available() or not dist.is_initialized():
                    raise RuntimeError("6.8B fsdp2_zero3 requires torchrun before construct")
        self.config = config
        self.root = root
        seed_everything(config.training.seed)
        self.device = training_device(config.training.device)
        self.dtype = resolve_dtype(config.training.precision)
        # Weights stay fp32. AMP downcasts CUDA matmuls. Feeding bf16 tensors
        # into fp32 Linear raises: mat1 BFloat16 / mat2 Float.
        self._amp_enabled = self.device.type == "cuda" and self.dtype in (torch.bfloat16, torch.float16)
        self.compute_dtype = torch.float32 if self._amp_enabled else self.dtype
        self.system = MinakanushiSystem(config.architecture).to(self.device)
        if train.activation_checkpoint:
            self.system.world_core.activation_checkpoint = True
        if train.parallelism == "fsdp2_zero3":
            from minakanushi.training.parallel import wrap_fsdp2

            self.system = wrap_fsdp2(
                self.system,
                activation_checkpoint=bool(train.activation_checkpoint),
            )
        if self._amp_enabled:
            self.system.world_core.checkpoint_amp_dtype = self.dtype
        self.constructor = StateConstructor(config.architecture)
        self.opt = None
        self.scheduler = None
        if not eval_only:
            self.opt = torch.optim.AdamW(
                self.system.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
                foreach=False,
            )
            self.scheduler = WarmupScheduler(self.opt, train.warmup_steps, train.learning_rate)
        self.constraints = MinakanushiConstraintKernel(config.simulation)
        self.policy = ActionPolicy()
        self._last_forward_s = 0.0
        self._last_backward_s = 0.0
        self.start_step = 1
        self.dataset = None
        self.heldout = None
        self.sampler = None
        if str(train.dataset_root).strip():
            data_root = Path(train.dataset_root)
            if not data_root.is_absolute():
                data_root = self.root / data_root
            self.dataset = JsonEpisodeDataset(
                data_root, seed=train.seed, split=str(train.dataset_split or "")
            )
            held_index = data_root / "heldout" / "index.jsonl"
            if held_index.is_file() and held_index.read_text(encoding="utf-8").strip():
                self.heldout = JsonEpisodeDataset(data_root, seed=train.seed, split="heldout")
            if train.sampler_mode != "uniform":
                self.sampler = PhaseCurriculumSampler(
                    self.dataset.paths, self.dataset.phases, seed=train.seed
                )
        self.dataset_cursor = 0
        self._resume_extras: dict = {}

    def resume_from(self, path: Path) -> None:
        state = apply_resume(path, self.system, self.opt, self.scheduler)
        self.start_step = int(state.last_step) + 1
        self.dataset_cursor = int(state.dataset_cursor)
        self._resume_extras = dict(state.extras)

    def _sampler_mode(self, step: int) -> str:
        mode = str(self.config.training.sampler_mode)
        if mode == "auto":
            job = int(step) - int(self.start_step) + 1
            return mode_for_job_step(job, warm_steps=self.config.training.warm_steps)
        return mode

    def _load_episode(
        self,
        step: int,
        *,
        scenario: str | None,
        episode_index: int | None,
        seed: int | None = None,
        length: int | None = None,
    ):
        train = self.config.training
        arch = self.config.architecture
        if self.dataset is not None:
            if scenario is not None:
                return self.dataset.episode_for_scenario(scenario, int(episode_index or 0))
            if episode_index is not None:
                return self.dataset.episode(int(episode_index))
            mode = self._sampler_mode(step)
            if self.sampler is not None and mode in {"warm", "intelligence"}:
                idx = self.sampler.choose(step, mode)
            else:
                idx = (step - 1) % len(self.dataset)
            self.dataset_cursor = int(idx)
            return self.dataset.episode(idx)
        ep_idx = int(episode_index) if episode_index is not None else (step - 1) % max(train.n_overfit_episodes, 1)
        scenario_name = scenario or TRAIN_CURRICULUM[ep_idx % len(TRAIN_CURRICULUM)]
        return generate_episode(
            self.config.simulation,
            seed=int(seed) if seed is not None else train.seed,
            episode_index=ep_idx,
            length=int(length) if length is not None else train.sequence_length,
            horizon=arch.prediction_horizons.short,
            scenario=scenario_name,
        )

    def _amp(self):
        if self._amp_enabled:
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        return nullcontext()

    def _encode(self, obs, episode_pos: float) -> MinaUnitBatch:
        units = self.system.perception.encode(obs, device=self.device, dtype=self.compute_dtype)
        now = obs.arrival_time if obs.arrival_time is not None else obs.timestamp
        return pack_units(
            units,
            batch_index=0,
            max_units=self.config.architecture.max_observations,
            latent_dim=self.config.architecture.latent_dim,
            episode_position=episode_pos,
            now=now,
            device=self.device,
            dtype=self.compute_dtype,
        )

    def _core_step(self, packed, world, live_writes):
        hints = self.system.memory.hints(world, live_writes=live_writes)
        pos = self.system.position_units(packed)
        fused = packed.semantic_embedding + pos.embedding
        constructed = self.constructor.apply(packed, world, fused, memory_hints=hints)
        _, core = self.system.observe_to_core(packed, constructed, hints)
        return pos, hints, constructed, core

    def unroll(
        self,
        step: int,
        *,
        scenario: str | None = None,
        episode_index: int | None = None,
        seed: int | None = None,
        length: int | None = None,
        episode=None,
    ) -> UnrollPacket:
        train = self.config.training
        arch = self.config.architecture
        episode = episode or self._load_episode(
            step, scenario=scenario, episode_index=episode_index, seed=seed, length=length
        )
        ep_idx = int(episode.episode_index)
        idx = training_frame(episode.scenario, len(episode.observations))
        obs = episode.observations[idx]
        obs_n = episode.observations[idx + 1]
        truth = episode.truth[idx]
        truth_n = episode.truth[idx + 1]

        world = empty_world_state(arch, 1, device=self.device, dtype=self.compute_dtype)
        world.entity_xy[0, 0] = torch.tensor(
            self.config.simulation.agent_start, device=self.device, dtype=self.compute_dtype
        )
        writes = None
        if idx > 0:
            with torch.no_grad():
                for t in range(idx):
                    packed_t = self._encode(episode.observations[t], float(t))
                    _, _, _, core_t = self._core_step(packed_t, world, live_writes=writes)
                    world = core_t.world_state
                    writes = core_t.memory_write_candidates

        before_xy = world.entity_xy.detach().clone()
        occupied_before = world.occupied.clone()
        packed = self._encode(obs, float(idx))
        evidence_xy, evidence_vel, has_evidence = evidence_for_slots(world, packed)
        should_revise = should_revise_mask(
            before_xy, evidence_xy, has_evidence, occupied_before, world.entity_id
        )
        pos, hints, constructed, core = self._core_step(packed, world, live_writes=writes)
        writes = core.memory_write_candidates
        pred = core.world_state

        packed_n = self._encode(obs_n, float(idx + 1))
        _, _, _, core_n = self._core_step(packed_n, pred, live_writes=writes)
        pred_n = core_n.world_state

        max_e = arch.world_slots
        true_ids, true_xy, true_vel, _ = _truth_tensors(truth, max_e, self.device, self.compute_dtype)
        aligned_xy, aligned_vel, aligned_occ = _align(pred.entity_id, pred.occupied, true_ids, true_xy, true_vel)
        true_ids_n, true_next, true_next_vel, _ = _truth_tensors(truth_n, max_e, self.device, self.compute_dtype)
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

        agent_xy = tuple(float(x) for x in episode.observations[idx].agent_xy)
        cand = StrategyCandidate(truth.action.lower(), truth.action, truth.action_target, 0.0, 0.0)
        alt = counterfactual_candidate(truth, agent_xy, self.config.simulation)
        if cand.strategy_id == alt.strategy_id:
            raise RuntimeError(
                f"counterfactual collapsed to labeled strategy {cand.strategy_id!r} action={truth.action!r}"
            )
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
            unobserved_mask=(pred.age_unobserved > 0) & pred.occupied,
            xy_std=pred.xy_std,
            existence=pred.existence,
            true_present=aligned_occ,
            hypothesized=pred.occupied,
            before_xy=before_xy,
            after_xy=pred.entity_xy,
            after_vel=pred.entity_vel,
            evidence_xy=evidence_xy,
            evidence_vel=evidence_vel,
            should_revise=should_revise,
            has_evidence=has_evidence,
            occupied_before=occupied_before,
            entity_id=world.entity_id,
        )
        assert_finite("loss.total", breakdown.total)
        return UnrollPacket(
            packed=packed,
            packed_n=packed_n,
            pos=pos,
            hints=hints,
            writes=writes,
            pred=pred,
            pred_n=pred_n,
            core=core,
            core_n=core_n,
            trajs=trajs,
            breakdown=breakdown,
            aligned_xy=aligned_xy,
            aligned_vel=aligned_vel,
            aligned_occ=aligned_occ,
            aligned_next=aligned_next,
            mem_mask=mem_mask,
            pred_future=pred_future,
            true_future=true_future,
            alt_future=alt_future,
            intra_b=intra_b,
            episode_index=ep_idx,
            frame_index=idx,
            scenario=episode.scenario,
            candidates=[cand, alt],
            obs_timestamp=float(obs.timestamp),
            obs_agent_xy=tuple(float(x) for x in obs.agent_xy),
            obs_agent_vel=tuple(float(x) for x in obs.agent_vel),
            before_xy=before_xy,
            evidence_xy=evidence_xy,
            should_revise=should_revise,
            has_evidence=has_evidence,
            occupied_before=occupied_before,
            n_constructor_corrections=len(constructed.corrections),
        )

    def _metrics(self, pkt: UnrollPacket) -> dict[str, float]:
        pred = pkt.pred
        pred_n = pkt.pred_n
        persist_den = 0
        persist_hit = 0
        reacq_den = 0
        reacq_hit = 0
        ids_t = {int(x) for x in pred.entity_id[0, pred.occupied[0]].tolist()} if bool(pred.occupied.any()) else set()
        ids_n = {int(x) for x in pred_n.entity_id[0, pred_n.occupied[0]].tolist()} if bool(pred_n.occupied.any()) else set()
        for eid in ids_t:
            persist_den += 1
            if eid in ids_n:
                persist_hit += 1
        for eid in ids_n:
            reacq_den += 1
            if eid in ids_t:
                reacq_hit += 1
        zeros = torch.zeros_like(pkt.writes)
        with torch.no_grad():
            with self._amp():
                _, _, _, core_off = self._core_step(pkt.packed_n, pkt.pred, live_writes=zeros)
        mem_slots = pkt.mem_mask if bool(pkt.mem_mask.any()) else pkt.pred_n.occupied
        mem_delta = float(
            memory_effect_delta(
                pkt.pred_n.latent_state,
                core_off.world_state.latent_state,
                mem_slots,
            ).detach()
        )
        err_with = masked_mse(pkt.pred_n.entity_xy, pkt.aligned_next, pkt.aligned_occ)
        err_without = masked_mse(core_off.world_state.entity_xy, pkt.aligned_next, pkt.aligned_occ)
        memory_future = float((err_without - err_with).detach())
        trajs_off = self.system.future.predict(
            core_off.world_state, pkt.candidates, max_horizon=self.config.architecture.prediction_horizons.short
        )
        primary_off = [t for t in trajs_off if t.strategy_id == pkt.candidates[0].strategy_id]
        pred_future_off = primary_off[0].states_xy.unsqueeze(0)
        ade_on, fde_on = displacement_error(pkt.pred_future, pkt.true_future, pkt.aligned_occ)
        ade_off, fde_off = displacement_error(pred_future_off, pkt.true_future, pkt.aligned_occ)
        memory_ade_on = float(ade_on.detach())
        memory_ade_off = float(ade_off.detach())
        memory_fde_on = float(fde_on.detach())
        memory_fde_off = float(fde_off.detach())
        memory_helps_future = 1.0 if memory_ade_on < memory_ade_off else 0.0
        primary = [t for t in pkt.trajs if t.strategy_id == pkt.candidates[0].strategy_id]
        branch_xy = torch.stack([t.states_xy for t in primary], dim=0)
        future_div = float(branch_diversity(branch_xy).detach())
        counterfactual = float(
            counterfactual_separation_score(pkt.pred_future[0, -1], pkt.alt_future[0, -1]).detach()
        )
        true_term = pkt.true_future[0, -1]
        best = torch.linalg.vector_norm(branch_xy[:, -1] - true_term, dim=-1).min(dim=0).values
        const = torch.linalg.vector_norm(pkt.aligned_xy[0] - true_term, dim=-1)
        occ = pkt.aligned_occ[0].to(best.dtype)
        coverage = float(((best < const).to(best.dtype) * occ).sum() / occ.sum().clamp_min(1.0))
        violations, loop_ok = policy_firewall_metrics(
            self.constraints,
            self.policy,
            pkt.candidates,
            group_by_strategy(pkt.trajs),
            self.config.simulation,
            self.config.simulation.home,
            pkt.obs_timestamp,
            pkt.obs_agent_xy,
            pkt.obs_agent_vel,
        )
        bundle = assemble_bundle(
            pred_xy=pred.entity_xy,
            true_xy=pkt.aligned_xy,
            pred_vel=pred.entity_vel,
            true_vel=pkt.aligned_vel,
            occupied=pkt.aligned_occ,
            pred_future=pkt.pred_future,
            true_future=pkt.true_future,
            persist_hits=persist_hit / max(persist_den, 1),
            reacquire_hits=reacq_hit / max(reacq_den, 1),
            uncertainty=pred.uncertainty,
            position_error=pred.entity_xy - pkt.aligned_xy,
            branch_xy=branch_xy,
            memory_delta=mem_delta,
            constraint_violations=violations,
            closed_loop_success=loop_ok,
            coverage=coverage,
            before_xy=pkt.before_xy,
            after_xy=pred.entity_xy,
            evidence_xy=pkt.evidence_xy,
            should_revise=pkt.should_revise,
            has_evidence=pkt.has_evidence,
            occupied_before=pkt.occupied_before,
            entity_id=pred.entity_id,
            memory_future_delta=memory_future,
            future_diversity=future_div,
            counterfactual_quality=counterfactual,
            memory_ade_on=memory_ade_on,
            memory_ade_off=memory_ade_off,
            memory_fde_on=memory_fde_on,
            memory_fde_off=memory_fde_off,
            memory_helps_future=memory_helps_future,
        )
        return asdict(bundle)

    def step_once(self, step: int) -> TrainLog:
        t0 = perf_counter()
        with self._amp():
            pkt = self.unroll(step)
        self._last_forward_s = perf_counter() - t0
        t1 = perf_counter()
        self.opt.zero_grad(set_to_none=True)
        pkt.breakdown.total.backward()
        grad_norm = clip_grad_norm_mixed(self.system.parameters(), self.config.training.grad_clip)
        self.opt.step()
        self.scheduler.step()
        self._last_backward_s = perf_counter() - t1
        traj_error = float(
            ((pkt.pred_future[:, -1] - pkt.true_future[:, -1]).pow(2) * pkt.aligned_occ.unsqueeze(-1)).sum().item()
        )
        metrics = None
        if step == 1 or step % self.config.training.eval_every == 0:
            metrics = self._metrics(pkt)
            if self.heldout is not None:
                with torch.no_grad():
                    held_ep = self.heldout.episode((step - 1) % len(self.heldout))
                    held_pkt = self.unroll(step, episode=held_ep)
                    held = self._metrics(held_pkt)
                metrics["heldout_score"] = float(held["future_ADE"])
            else:
                metrics["heldout_score"] = None
        return TrainLog(
            step=step,
            loss=float(pkt.breakdown.total.item()),
            terms={k: float(v.item()) for k, v in pkt.breakdown.terms.items()},
            grad_norm=grad_norm,
            traj_error=traj_error,
            metrics=metrics,
        )

    def fit(self, out_dir: Path, *, resume: Path | None = None) -> list[TrainLog]:
        train = self.config.training
        if resume is not None:
            self.resume_from(Path(resume))
        logs: list[TrainLog] = []
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = out_dir / "metrics.jsonl"
        abort_reason = None
        start = int(self.start_step)
        end = start + int(train.steps) - 1
        for step in range(start, end + 1):
            log = self.step_once(step)
            logs.append(log)
            if step % train.log_every == 0 or step == start:
                if is_rank0():
                    print(
                        f"step={step} loss={log.loss:.4f} traj_err={log.traj_error:.4f} "
                        f"grad={log.grad_norm:.4f} fwd={self._last_forward_s:.3f}s "
                        f"bwd={self._last_backward_s:.3f}s terms={log.terms}",
                        flush=True,
                    )
            if log.metrics is not None and is_rank0():
                eval_row = {
                    "step": step,
                    "loss": log.loss,
                    "future_ADE": log.metrics.get("future_ADE"),
                    "future_FDE": log.metrics.get("future_FDE"),
                    "revision_accuracy": log.metrics.get("revision_accuracy"),
                    "false_revision": log.metrics.get("false_revision_rate"),
                    "memory_future_delta": log.metrics.get("memory_future_delta"),
                    "counterfactual_distance": log.metrics.get("counterfactual_quality"),
                    "heldout_score": log.metrics.get("heldout_score"),
                    "terms": log.terms,
                }
                with metrics_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"step": step, "loss": log.loss, "terms": log.terms, "metrics": log.metrics}) + "\n")
                with (out_dir / "experiment.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(eval_row) + "\n")
                print(
                    f"eval step={step} loss={log.loss:.4f} ADE={eval_row['future_ADE']} "
                    f"FDE={eval_row['future_FDE']} rev={eval_row['revision_accuracy']} "
                    f"false_rev={eval_row['false_revision']} memΔ={eval_row['memory_future_delta']} "
                    f"cf={eval_row['counterfactual_distance']} heldout={eval_row['heldout_score']}",
                    flush=True,
                )
            abort_reason = _diagnose_run(logs)
            if abort_reason is not None:
                if is_rank0():
                    print(f"ABORT {abort_reason}", flush=True)
                break
            if step % train.checkpoint_every == 0 or step == end:
                gathered = collect_full_checkpoint(self.system, self.opt)
                save_mina(
                    out_dir / f"minakanushi_stage{train.stage}_step{step}.mina",
                    self.system,
                    optimizer=self.opt,
                    shard_max_bytes=int(train.shard_max_bytes),
                    gathered=gathered,
                    extras={
                        "stage": train.stage,
                        "step": step,
                        "seed": train.seed,
                        "dataset_name": train.dataset_name,
                        "dataset_cursor": int(self.dataset_cursor if self.dataset is not None else step),
                        "dataset_root": train.dataset_root,
                        "sampler_mode": train.sampler_mode,
                        "warm_steps": train.warm_steps,
                        "identity_initialized": True,
                        "identity_trainable": False,
                        "identity": canonical_identity_payload()["identity_state"],
                        "scheduler": self.scheduler.state_dict(),
                        "loss": log.loss,
                        "traj_error": log.traj_error,
                        "metrics": {"traj_error": log.traj_error, "loss": log.loss},
                    },
                    tensors={"rng": capture_rng(), "scheduler": self.scheduler.state_dict()},
                )
        last = logs[-1]
        if is_rank0():
            torch.save(
                {
                    "seed": train.seed,
                    "n_overfit_episodes": train.n_overfit_episodes,
                    "sequence_length": train.sequence_length,
                    "step": last.step,
                    "loss": last.loss,
                    "entity_xy": None,
                },
                out_dir / "reference_meta.pt",
            )
        # Frozen inference snapshot from a deterministic episode for reload comparison.
        with torch.no_grad():
            self.system.eval()
            with self._amp():
                pkt = self.unroll(1)
            if is_rank0():
                torch.save(
                    {
                        "seed": train.seed,
                        "episode_index": pkt.episode_index,
                        "frame_index": pkt.frame_index,
                        "entity_xy": pkt.pred.entity_xy.detach().cpu(),
                        "latent_state": pkt.pred.latent_state.detach().cpu(),
                        "future_terminal": pkt.pred_future[:, -1].detach().cpu(),
                    },
                    out_dir / "reference_inference.pt",
                )
            self.system.train()
        dist_barrier()
        if abort_reason is not None:
            raise RuntimeError(abort_reason)
        return logs


def _diagnose_run(logs: list[TrainLog]) -> str | None:
    last = logs[-1]
    if not (last.loss == last.loss) or last.loss == float("inf"):
        return "loss is NaN/Inf"
    if last.loss == 0.0 and last.step <= 5:
        return "loss collapsed to zero immediately"
    if last.grad_norm > 1e6:
        return "gradient norm diverges"
    if len(logs) >= 20:
        window = logs[:20]
        losses = [x.loss for x in window]
        if max(losses) - min(losses) < 1e-8:
            return "loss stays constant"
        if last.loss > 50.0 * logs[0].loss + 1.0:
            return "loss oscillates explosively"
        terms0 = logs[0].terms
        terms1 = last.terms
        if terms0:
            dominant = max(terms1, key=terms1.get)
            rest = sum(v for k, v in terms1.items() if k != dominant)
            if terms1[dominant] > 50.0 * max(rest, 1e-8) and abs(terms1[dominant] - terms0[dominant]) < 1e-6:
                return f"one loss dominates and is frozen: {dominant}"
    return None


def trainer_from_files(root: Path, training_yaml: Path) -> Trainer:
    from minakanushi.architecture.config import load_training
    from minakanushi.training.v031_dataset import assert_v031_train_dataset

    training = load_training(training_yaml)
    assert_v031_train_dataset(root, training)
    config = load_config(
        root / training.architecture,
        training_path=training_yaml,
        runtime_path=root / "configs" / "runtime" / "cpu.yaml",
        simulation_path=root / training.simulation,
    )
    return Trainer(config, root)