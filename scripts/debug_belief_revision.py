"""Trace why Stage A belief_revision_accuracy stayed 0.

Not a training run. Does not construct 6.8B.
"""

from __future__ import annotations

import inspect
from collections import Counter
from pathlib import Path

import torch

from minakanushi.architecture.config import load_config
from minakanushi.architecture.mina_unit import pack_units
from minakanushi.architecture.model import MinakanushiSystem
from minakanushi.state.constructor import StateConstructor, empty_world_state
from minakanushi.training.metrics import assemble_bundle, belief_revision_accuracy
from minakanushi.training.objectives import compute_objectives
from simulations.synthetic_world.dataset import GATE03_SCENARIOS, SCENARIOS, generate_episode
from simulations.synthetic_world.dataset_v1 import SPLIT_SCENARIOS, episode_to_record

ROOT = Path(__file__).resolve().parents[1]


def _slot(world, eid: int) -> int | None:
    for s in range(world.entity_id.shape[1]):
        if bool(world.occupied[0, s]) and int(world.entity_id[0, s].item()) == int(eid):
            return s
    return None


def _xy(world, eid: int) -> tuple[float, float] | None:
    s = _slot(world, eid)
    if s is None:
        return None
    return (float(world.entity_xy[0, s, 0]), float(world.entity_xy[0, s, 1]))


def _vel(world, eid: int) -> tuple[float, float] | None:
    s = _slot(world, eid)
    if s is None:
        return None
    return (float(world.entity_vel[0, s, 0]), float(world.entity_vel[0, s, 1]))


def encode(system, obs, device, dtype, episode_pos: float, max_obs: int, dim: int):
    units = system.perception.encode(obs, device=device, dtype=dtype)
    now = obs.arrival_time if obs.arrival_time is not None else obs.timestamp
    return pack_units(
        units,
        batch_index=0,
        max_units=max_obs,
        latent_dim=dim,
        episode_position=episode_pos,
        now=now,
        device=device,
        dtype=dtype,
    )


def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def trainer_sampling() -> None:
    print_section("1. What the trainer actually samples")
    print("generate_episode() without scenario uses SCENARIOS[episode_index % len]")
    print("SCENARIOS:", SCENARIOS)
    print("GATE03_SCENARIOS (not in SCENARIOS):", GATE03_SCENARIOS)
    n_overfit = 16
    counts = Counter(SCENARIOS[i % len(SCENARIOS)] for i in range(n_overfit))
    print("Stage A n_overfit_episodes=16 scenario counts:", dict(counts))
    print("hidden_correction in those 16:", counts.get("hidden_correction", 0))
    print("conflict in those 16:", counts.get("conflict", 0))
    thousand = Counter(SCENARIOS[i % len(SCENARIOS)] for i in range(1000))
    print("1000 trainer episode_index counts:")
    for name in (*SCENARIOS, *GATE03_SCENARIOS):
        print(f"  {name:20s} {thousand.get(name, 0)}")
    print()
    print("Dataset v1 splits (files, not used by trainer.unroll):")
    for split, names in SPLIT_SCENARIOS.items():
        print(f"  {split:16s} {names}")
    print("hidden_correction lives in OOD only. Train split never sees it.")
    print()
    print("trainer.unroll frame picker: idx = min(3, length-3)")
    print("hidden_correction hide window is t=1..5, reappear t=6.")
    print("Even if the scenario were sampled, idx=3 is still hidden — miss the revision.")


def metric_wiring() -> None:
    print_section("2. Metric wiring")
    params = list(inspect.signature(assemble_bundle).parameters)
    print("assemble_bundle params:", params)
    print("belief_revision_accuracy in assemble_bundle:", "belief_revision_accuracy" in params)
    dummy = torch.zeros(1, 2, 2)
    mask = torch.ones(1, 2, dtype=torch.bool)
    bundle = assemble_bundle(
        pred_xy=dummy,
        true_xy=dummy,
        pred_vel=dummy,
        true_vel=dummy,
        occupied=mask,
        pred_future=dummy.unsqueeze(1),
        true_future=dummy.unsqueeze(1),
        persist_hits=1.0,
        reacquire_hits=1.0,
        uncertainty=torch.zeros(1, 2, 8),
        position_error=dummy,
        branch_xy=torch.zeros(2, 1, 2, 2),
        memory_delta=1.6,
        constraint_violations=0,
        closed_loop_success=1.0,
        coverage=0.0,
    )
    print("assemble_bundle().belief_revision_accuracy =", bundle.belief_revision_accuracy)
    print("That is the dataclass default. Trainer._metrics never calls belief_revision_accuracy().")
    toward = belief_revision_accuracy(
        torch.tensor([[100.0, 0.0]]),
        torch.tensor([[110.0, 0.0]]),
        torch.tensor([[111.0, 0.0]]),
    )
    print("helper(old=100, new=110, evidence=111) =", float(toward), "(would be 1 if wired)")


def walk_hidden_correction() -> None:
    print_section("3. Debug episode: hidden_correction sequential constructor")
    cfg = load_config(
        ROOT / "configs" / "architecture" / "cpu_dev.yaml",
        runtime_path=ROOT / "configs" / "runtime" / "cpu.yaml",
        simulation_path=ROOT / "configs" / "simulation" / "milestone1.yaml",
    )
    device = torch.device("cpu")
    dtype = torch.float32
    arch = cfg.architecture
    system = MinakanushiSystem(arch)
    system.eval()
    ctor = StateConstructor(arch)
    ep = generate_episode(cfg.simulation, seed=7, episode_index=0, length=12, horizon=4, scenario="hidden_correction")
    rec = episode_to_record(ep)
    mover = None
    for i, kind in enumerate(ep.truth[0].kind):
        if str(kind) == "mover":
            mover = int(ep.truth[0].entity_id[i])
            break
    if mover is None:
        raise RuntimeError("hidden_correction episode has no mover")
    print("scenario", ep.scenario, "length", len(ep.observations), "mover_id", mover)
    print("teacher corrections (dataset_v1 outcomes):", rec["corrections"][:8])
    event_types = Counter(e["type"] for e in rec["events"])
    print("teacher events:", dict(event_types))

    world = empty_world_state(arch, 1, device=device, dtype=dtype)
    writes = None
    print()
    print(f"{'t':>3} {'visible':>8} {'gt_xy':>18} {'belief_xy':>18} {'belief_vel':>16} {'n_corr':>6} reasons")
    reappear_dump = None
    for t, obs in enumerate(ep.observations):
        visible = [int(v["id"]) for v in obs.visible]
        packed = encode(system, obs, device, dtype, float(t), arch.max_observations, arch.latent_dim)
        hints = system.memory.hints(world, live_writes=writes)
        pos = system.position_units(packed)
        fused = packed.semantic_embedding + pos.embedding
        before = world
        constructed = ctor.apply(packed, before, fused, memory_hints=hints)
        with torch.no_grad():
            _, core = system.observe_to_core(packed, constructed, hints)
        after_dwc = core.world_state
        writes = core.memory_write_candidates
        gt = None
        for i, eid in enumerate(ep.truth[t].entity_id):
            if int(eid) == mover:
                gt = (float(ep.truth[t].xy[i, 0]), float(ep.truth[t].xy[i, 1]))
        reasons = [ev.correction_reason for ev in constructed.corrections]
        print(
            f"{t:3d} {str(mover in visible):>8} {str(gt):>18} "
            f"{str(_xy(constructed, mover)):>18} {str(_vel(constructed, mover)):>16} "
            f"{len(constructed.corrections):6d} {reasons}"
        )
        if t == 6:
            reappear_dump = {
                "t": t,
                "visible": visible,
                "gt_xy": gt,
                "BELIEF_BEFORE_ctor": {
                    "xy": _xy(before, mover),
                    "vel": _vel(before, mover),
                    "age": None if _slot(before, mover) is None else float(before.age_unobserved[0, _slot(before, mover)]),
                    "existence": None if _slot(before, mover) is None else float(before.existence[0, _slot(before, mover)]),
                },
                "NEW_OBS": {
                    "visible_ids": visible,
                    "obs_xy": next(
                        ((float(v["xy"][0]), float(v["xy"][1])) for v in obs.visible if int(v["id"]) == mover),
                        None,
                    ),
                },
                "CORRECTION_EVENT": [
                    {
                        "entity_id": ev.entity_id,
                        "reason": ev.correction_reason,
                        "old_xy": ev.old_xy,
                        "new_xy": ev.new_xy,
                        "old_vel": ev.old_vel,
                        "new_vel": ev.new_vel,
                        "magnitude": ev.correction_magnitude,
                    }
                    for ev in constructed.corrections
                ],
                "BELIEF_AFTER_ctor": {"xy": _xy(constructed, mover), "vel": _vel(constructed, mover)},
                "BELIEF_AFTER_DWC": {"xy": _xy(after_dwc, mover), "vel": _vel(after_dwc, mover)},
                "dwc_delta_xy": None,
            }
            b = reappear_dump["BELIEF_AFTER_ctor"]["xy"]
            d = reappear_dump["BELIEF_AFTER_DWC"]["xy"]
            if b is not None and d is not None:
                reappear_dump["dwc_delta_xy"] = (d[0] - b[0], d[1] - b[1])
        world = after_dwc

    print_section("3b. Maga dump at reappear t=6")
    assert reappear_dump is not None
    for k, v in reappear_dump.items():
        print(f"{k}: {v}")

    print_section("3c. Trainer-style two-frame unroll (idx=3) on the SAME episode")
    idx = min(3, len(ep.observations) - 3)
    world = empty_world_state(arch, 1, device=device, dtype=dtype)
    packed = encode(system, ep.observations[idx], device, dtype, float(idx), arch.max_observations, arch.latent_dim)
    packed_n = encode(
        system, ep.observations[idx + 1], device, dtype, float(idx + 1), arch.max_observations, arch.latent_dim
    )
    hints = system.memory.hints(world, live_writes=None)
    pos = system.position_units(packed)
    fused = packed.semantic_embedding + pos.embedding
    constructed = ctor.apply(packed, world, fused, memory_hints=hints)
    with torch.no_grad():
        _, core = system.observe_to_core(packed, constructed, hints)
    pred = core.world_state
    hints_n = system.memory.hints(pred, live_writes=core.memory_write_candidates)
    pos_n = system.position_units(packed_n)
    fused_n = packed_n.semantic_embedding + pos_n.embedding
    constructed_n = ctor.apply(packed_n, pred, fused_n, memory_hints=hints_n)
    print("idx", idx, "visible_t", [int(v["id"]) for v in ep.observations[idx].visible])
    print("idx+1 visible", [int(v["id"]) for v in ep.observations[idx + 1].visible])
    print("corrections at t (first obs from empty):", len(constructed.corrections), [e.correction_reason for e in constructed.corrections])
    print("corrections at t+1:", len(constructed_n.corrections), [e.correction_reason for e in constructed_n.corrections])
    print("hypothesis_revision count t+1:", sum(1 for e in constructed_n.corrections if e.correction_reason == "hypothesis_revision"))
    print("This is the batch the Stage A trainer actually optimized — empty→frame3→frame4, not t=6.")


def loss_terms_on_reappear() -> None:
    print_section("4. Loss terms: L_belief is NLL(xy), not CorrectionEvent")
    from minakanushi.architecture.config import load_training

    train = load_training(ROOT / "configs" / "training" / "stage0_overfit.yaml")
    print("lambdas.belief", train.lambdas.belief, "lambdas.state", train.lambdas.state)
    dummy = torch.zeros(1, 4, 2)
    occ = torch.ones(1, 4, dtype=torch.bool)
    breakdown = compute_objectives(
        pred_xy=dummy,
        true_xy=dummy,
        occupied=occ,
        pred_next_xy=dummy,
        true_next_xy=dummy,
        pred_future_xy=dummy.unsqueeze(1),
        true_future_xy=dummy.unsqueeze(1),
        uncertainty=torch.ones(1, 4, 8) * 0.3,
        memory_xy=dummy,
        memory_true_xy=dummy,
        memory_mask=occ,
        causal_pred=dummy,
        causal_true=dummy,
        alt_future_xy=dummy.unsqueeze(1),
        intra_branch_xy=dummy.unsqueeze(1),
        latent=torch.zeros(1, 4, 64),
        training=train,
        xy_std=torch.ones(1, 4, 2) * 0.1,
        existence=torch.ones(1, 4),
        true_present=occ,
        hypothesized=occ,
    )
    print("compute_objectives.terms keys:", list(breakdown.terms))
    print("no term named revision / correction / hypothesis")
    print("L_belief = gaussian NLL(pred_xy, true_xy) + existence BCE")
    print("WorldState.corrections is never an argument to compute_objectives.")
    print()
    print("Stage A logged terms (from GPU run), not loss total:")
    print("  step 1   belief=149.12  state=0.76  future=3.27")
    print("  step 200 belief=-3.90   state=0.018 future=0.21")
    print("Belief term moved because XY NLL moved. That is tracking, not revision.")


def main() -> None:
    trainer_sampling()
    metric_wiring()
    walk_hidden_correction()
    loss_terms_on_reappear()
    print_section("BREAK POINTS")
    print("A. Metric always 0: assemble_bundle does not compute it.")
    print("B. Train batch never contains hidden_correction/conflict (SCENARIOS + idx=3).")
    print("C. Loss never sees CorrectionEvent; L_belief is current-frame XY NLL.")
    print("D. DWC xy residual can overwrite constructor revision before loss.")
    print("Do not scale. Next commit: connect revision objective to the training signal.")


if __name__ == "__main__":
    main()
