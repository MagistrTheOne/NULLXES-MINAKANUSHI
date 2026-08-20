"""Post-overfit Gate 02 checks: reload, firewalls, memory, uncertainty, baselines."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from minakanushi.architecture.config import load_config
from minakanushi.architecture.model import MinakanushiSystem
from minakanushi.core.recurrent_state import clone_world
from minakanushi.future.engine import group_by_strategy
from minakanushi.runtime.engine import MinakanushiEngine
from minakanushi.strategy.candidate import StrategyCandidate
from minakanushi.training.baselines import constant_position, constant_velocity, no_memory_state, single_future
from minakanushi.training.checkpoint import latest_mina, load_mina
from minakanushi.training.metrics import displacement_error, masked_mse
from minakanushi.training.trainer import trainer_from_files
from minakanushi.utils.seed import seed_everything
from simulations.synthetic_world.dataset import generate_episode
from simulations.synthetic_world.world import SyntheticWorld


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "stage0_overfit"


def _latest_mina() -> Path:
    return latest_mina(OUT)


def _fresh_system():
    cfg = load_config(
        ROOT / "configs" / "architecture" / "cpu_dev.yaml",
        training_path=ROOT / "configs" / "training" / "stage0_overfit.yaml",
        runtime_path=ROOT / "configs" / "runtime" / "cpu.yaml",
        simulation_path=ROOT / "configs" / "simulation" / "milestone1.yaml",
    )
    system = MinakanushiSystem(cfg.architecture)
    return cfg, system


def reload_check() -> dict:
    path = _latest_mina()
    cfg, system = _fresh_system()
    manifest = load_mina(path, system)
    ref = torch.load(OUT / "reference_inference.pt", map_location="cpu")
    trainer = trainer_from_files(ROOT, ROOT / "configs" / "training" / "stage0_overfit.yaml")
    # Replace trainer system with freshly loaded weights (different object).
    trainer.system.load_state_dict(system.state_dict(), strict=True)
    trainer.system.eval()
    with torch.no_grad():
        pkt = trainer.unroll(1)
    xy_err = float((pkt.pred.entity_xy.cpu() - ref["entity_xy"]).abs().max())
    lat_err = float((pkt.pred.latent_state.cpu() - ref["latent_state"]).abs().max())
    fut_err = float((pkt.pred_future[:, -1].cpu() - ref["future_terminal"]).abs().max())
    ok = xy_err < 1e-5 and lat_err < 1e-5 and fut_err < 1e-5
    return {
        "path": str(path),
        "architecture_version": manifest.get("architecture_version"),
        "latent_dim": manifest.get("latent_dim"),
        "strict_load": True,
        "fresh_process_object": True,
        "xy_max_err": xy_err,
        "latent_max_err": lat_err,
        "future_max_err": fut_err,
        "pass": ok,
    }


def future_firewall() -> dict:
    engine = MinakanushiEngine(
        load_config(
            ROOT / "configs" / "architecture" / "cpu_dev.yaml",
            runtime_path=ROOT / "configs" / "runtime" / "cpu.yaml",
            simulation_path=ROOT / "configs" / "simulation" / "milestone1.yaml",
        )
    )
    path = _latest_mina()
    load_mina(path, engine.system)
    world = SyntheticWorld(engine.config.simulation, seed=4)
    state = engine.initialize()
    obs = world.observe()
    result = engine.step(obs, state)
    w = result.state.world
    before = {
        "xy_ptr": w.entity_xy.data_ptr(),
        "lat_ptr": w.latent_state.data_ptr(),
        "occ_ptr": w.occupied.data_ptr(),
        "xy": w.entity_xy.clone(),
        "lat": w.latent_state.clone(),
        "occ": w.occupied.clone(),
    }
    cand = StrategyCandidate("move", "MOVE_TO", (8.0, 2.0), 0.0, 0.0)
    futures = engine.future.predict(w, [cand], max_horizon=4)
    after_ptr = (w.entity_xy.data_ptr(), w.latent_state.data_ptr(), w.occupied.data_ptr())
    mutated = bool((w.entity_xy - before["xy"]).abs().sum() > 0) or bool((w.latent_state - before["lat"]).abs().sum() > 0)
    aliased = after_ptr != (before["xy_ptr"], before["lat_ptr"], before["occ_ptr"])
    return {
        "ptr_unchanged": not aliased,
        "values_unchanged": not mutated,
        "n_futures": len(futures),
        "pass": (not aliased) and (not mutated) and len(futures) > 0,
    }


def constraint_firewall() -> dict:
    engine = MinakanushiEngine(
        load_config(
            ROOT / "configs" / "architecture" / "cpu_dev.yaml",
            runtime_path=ROOT / "configs" / "runtime" / "cpu.yaml",
            simulation_path=ROOT / "configs" / "simulation" / "milestone1.yaml",
        )
    )
    path = _latest_mina()
    load_mina(path, engine.system)
    world = SyntheticWorld(engine.config.simulation, seed=1)
    state = engine.initialize()
    obs = world.observe()
    result = engine.step(obs, state)
    forbidden = StrategyCandidate("raid_restricted", "MOVE_TO", (8.2, 8.2), 100.0, 0.0)
    safe = StrategyCandidate("hold", "SAFE_HOLD", (float(world.agent.xy[0]), float(world.agent.xy[1])), -10.0, 0.0)
    futures = engine.future.predict(result.state.world, [forbidden, safe], max_horizon=8)
    allowed, rejected, audits = engine.constraints.filter([forbidden, safe], group_by_strategy(futures))
    intent = engine.policy.select(allowed, group_by_strategy(futures), engine.config.simulation.home, obs.timestamp)
    a_rejected = forbidden.strategy_id in {c.strategy_id for c in rejected}
    a_not_allowed = forbidden.strategy_id not in {a.strategy_id for a in allowed}
    return {
        "A_rejected": a_rejected,
        "A_not_in_allowed": a_not_allowed,
        "intent": intent.strategy_id,
        "intent_not_A": intent.strategy_id != "raid_restricted",
        "constraint_violation_count": 0 if a_rejected and intent.strategy_id != "raid_restricted" else 1,
        "pass": a_rejected and a_not_allowed and intent.strategy_id != "raid_restricted",
    }


def branching() -> dict:
    trainer = trainer_from_files(ROOT, ROOT / "configs" / "training" / "stage0_overfit.yaml")
    load_mina(_latest_mina(), trainer.system)
    trainer.system.eval()
    with torch.no_grad():
        pkt = trainer.unroll(1)
    strategy_id = pkt.trajs[0].strategy_id
    branches = [t for t in pkt.trajs if t.strategy_id == strategy_id]
    recs = []
    for t in branches:
        recs.append(
            {
                "branch_id": t.branch_id,
                "probability": float(t.probability.detach()),
                "uncertainty": float(t.uncertainty.detach()),
                "terminal_agent": [float(x) for x in t.terminal_xy[0].detach()],
                "prob_eq_unc": abs(float(t.probability) - float(t.uncertainty)) < 1e-8,
            }
        )
    identical = True
    for i in range(len(branches)):
        for j in range(i + 1, len(branches)):
            if float((branches[i].states_xy - branches[j].states_xy).abs().max()) > 1e-8:
                identical = False
    return {
        "strategy": strategy_id,
        "branches": recs,
        "probability_ne_uncertainty": all(not r["prob_eq_unc"] for r in recs),
        "branches_not_identical": not identical,
        "pass": all(not r["prob_eq_unc"] for r in recs) and (not identical),
    }


def memory_occlusion() -> dict:
    trainer = trainer_from_files(ROOT, ROOT / "configs" / "training" / "stage0_overfit.yaml")
    load_mina(_latest_mina(), trainer.system)
    trainer.system.eval()
    arch = trainer.config.architecture
    episode = generate_episode(
        trainer.config.simulation,
        seed=trainer.config.training.seed,
        episode_index=3,
        length=trainer.config.training.sequence_length,
        horizon=arch.prediction_horizons.short,
        scenario="occlusion",
    )
    idx = min(3, len(episode.observations) - 3)
    from minakanushi.state.constructor import empty_world_state

    world = empty_world_state(arch, 1, device=trainer.device, dtype=trainer.dtype)
    packed = trainer._encode(episode.observations[idx], float(idx))
    pos, hints, core = trainer._core_step(packed, world, live_writes=None)
    writes = core.memory_write_candidates
    packed_n = trainer._encode(episode.observations[idx + 1], float(idx + 1))
    _, _, on = trainer._core_step(packed_n, core.world_state, live_writes=writes)
    zeros = torch.zeros_like(writes)
    _, _, off = trainer._core_step(packed_n, core.world_state, live_writes=zeros)
    occ = on.world_state.occupied
    delta = float(((on.world_state.entity_xy - off.world_state.entity_xy).pow(2) * occ.unsqueeze(-1).to(on.world_state.entity_xy.dtype)).mean())
    persist_on = int(on.world_state.occupied.sum())
    persist_off = int(off.world_state.occupied.sum())
    return {
        "memory_effect_delta": delta,
        "occupied_with_memory": persist_on,
        "occupied_without_memory": persist_off,
        "pass": delta > 1e-8,
    }


def uncertainty_order() -> dict:
    trainer = trainer_from_files(ROOT, ROOT / "configs" / "training" / "stage0_overfit.yaml")
    load_mina(_latest_mina(), trainer.system)
    trainer.system.eval()
    means = {}
    for name in ("const_velocity", "noisy", "missing", "delayed"):
        ep_idx = {"const_velocity": 0, "noisy": 4, "missing": 5, "delayed": 7}[name]
        # unroll uses episode_index from step; force via generate inside custom
        pkt_means = []
        with torch.no_grad():
            episode = generate_episode(
                trainer.config.simulation,
                seed=trainer.config.training.seed,
                episode_index=ep_idx,
                length=trainer.config.training.sequence_length,
                horizon=trainer.config.architecture.prediction_horizons.short,
                scenario=name,
            )
            from minakanushi.state.constructor import empty_world_state

            world = empty_world_state(trainer.config.architecture, 1, device=trainer.device, dtype=trainer.dtype)
            packed = trainer._encode(episode.observations[3], 3.0)
            _, _, core = trainer._core_step(packed, world, live_writes=None)
            u = core.world_state.uncertainty[0, core.world_state.occupied[0]].mean()
            pkt_means.append(float(u))
        means[name] = sum(pkt_means) / len(pkt_means)
    clean = means["const_velocity"]
    degraded = (means["noisy"] + means["missing"] + means["delayed"]) / 3.0
    return {"per_scenario": means, "clean": clean, "degraded": degraded, "pass": clean < degraded}


def baselines() -> dict:
    trainer = trainer_from_files(ROOT, ROOT / "configs" / "training" / "stage0_overfit.yaml")
    load_mina(_latest_mina(), trainer.system)
    trainer.system.eval()
    dt = trainer.config.architecture.dt
    h = trainer.config.architecture.prediction_horizons.short
    model_ade = []
    cp_ade = []
    cv_ade = []
    sf_ade = []
    with torch.no_grad():
        for ep in range(16):
            pkt = trainer.unroll(ep + 1)
            ade_m, _ = displacement_error(pkt.pred_future, pkt.true_future, pkt.aligned_occ)
            model_ade.append(float(ade_m))
            cp = constant_position(pkt.aligned_xy, h)
            cv = constant_velocity(pkt.aligned_xy, pkt.aligned_vel, dt, h)
            sf = single_future(pkt.aligned_xy, pkt.aligned_vel, dt, h)
            a_cp, _ = displacement_error(cp, pkt.true_future, pkt.aligned_occ)
            a_cv, _ = displacement_error(cv, pkt.true_future, pkt.aligned_occ)
            a_sf, _ = displacement_error(sf, pkt.true_future, pkt.aligned_occ)
            cp_ade.append(float(a_cp))
            cv_ade.append(float(a_cv))
            sf_ade.append(float(a_sf))
    def mean(xs):
        return sum(xs) / max(len(xs), 1)
    return {
        "model_future_ADE": mean(model_ade),
        "constant_position_ADE": mean(cp_ade),
        "constant_velocity_ADE": mean(cv_ade),
        "single_future_ADE": mean(sf_ade),
        "beats_constant_position": mean(model_ade) < mean(cp_ade),
    }


def _rss_bytes() -> int | None:
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
        GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        if GetProcessMemoryInfo(GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
    except Exception:
        return None
    return None


def profile() -> dict:
    trainer = trainer_from_files(ROOT, ROOT / "configs" / "training" / "stage0_overfit.yaml")
    load_mina(_latest_mina(), trainer.system)
    n = trainer.system.parameter_report()
    ckpt = _latest_mina()
    ram = _rss_bytes()
    from time import perf_counter

    t0 = perf_counter()
    pkt = trainer.unroll(1)
    fwd = perf_counter() - t0
    t1 = perf_counter()
    trainer.opt.zero_grad(set_to_none=True)
    pkt.breakdown.total.backward()
    bwd = perf_counter() - t1
    return {
        "parameter_count": n,
        "checkpoint_bytes": ckpt.stat().st_size,
        "checkpoint_path": str(ckpt),
        "rss_bytes": ram,
        "peak_working_set_bytes": ram,
        "forward_s": fwd,
        "backward_s": bwd,
        "step_s": fwd + bwd,
    }


def main() -> None:
    seed_everything(7)
    report = {
        "reload": reload_check(),
        "future_firewall": future_firewall(),
        "constraint_firewall": constraint_firewall(),
        "branching": branching(),
        "memory": memory_occlusion(),
        "uncertainty": uncertainty_order(),
        "baselines": baselines(),
        "profile": profile(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=True))
    (OUT / "gate02_post.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
