"""FSDP2 resume must remap integer Adam ids to FQNs. Not a weights-only clone."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from minakanushi.training.optimizer_state import normalize_optimizer_state, optimizer_param_fqns
from minakanushi.training.parallel import apply_full_checkpoint


def _tiny() -> tuple[nn.Module, torch.optim.AdamW]:
    module = nn.Sequential(nn.Linear(4, 4, bias=True), nn.Linear(4, 2, bias=True))
    opt = torch.optim.AdamW(module.parameters(), lr=1e-3, foreach=False)
    loss = module(torch.ones(3, 4)).sum()
    loss.backward()
    opt.step()
    return module, opt


def test_normalize_integer_nested_state_uses_fqns() -> None:
    module, opt = _tiny()
    fqns = optimizer_param_fqns(module, opt)
    n = len(fqns)
    payload = {
        "state": {
            i: {
                "step": torch.tensor(64),
                "exp_avg": torch.zeros_like(param),
                "exp_avg_sq": torch.zeros_like(param),
            }
            for i, param in enumerate(module.parameters())
        },
        "param_groups": [{"lr": 1e-3, "params": list(range(n))}],
    }
    out = normalize_optimizer_state(module, opt, payload)
    assert set(out["state"]) == set(fqns)
    assert 28 not in out["state"]
    assert "28" not in out["state"]
    assert out["param_groups"][0]["params"] == list(fqns)
    assert int(out["state"][fqns[0]]["step"].item()) == 64


def test_normalize_flat_state_28_step_keys() -> None:
    module, opt = _tiny()
    fqns = optimizer_param_fqns(module, opt)
    flat: dict[str, object] = {
        "param_groups.0.lr": 1e-3,
        "param_groups.0.weight_decay": 0.01,
    }
    for i, param in enumerate(module.parameters()):
        flat[f"param_groups.0.params.{i}"] = i
        flat[f"state.{i}.step"] = torch.tensor(64)
        flat[f"state.{i}.exp_avg"] = torch.zeros_like(param)
        flat[f"state.{i}.exp_avg_sq"] = torch.zeros_like(param)
    # The H200 failure mode: FSDP unflatten looks up this exact key in an FQN map.
    assert "state.28.step" not in flat
    flat["state.28.step"] = torch.tensor(64)
    try:
        normalize_optimizer_state(module, opt, flat)
        raise AssertionError("index 28 must fail loud on a 4-parameter net")
    except ValueError as exc:
        assert "28" in str(exc)
    del flat["state.28.step"]
    out = normalize_optimizer_state(module, opt, flat)
    assert "state.28.step" not in out["state"]
    assert fqns[0] in out["state"]


def test_fqn_state_passes_through() -> None:
    module, opt = _tiny()
    fqns = optimizer_param_fqns(module, opt)
    payload = {
        "state": {fqns[0]: {"step": torch.tensor(7)}},
        "param_groups": [{"params": list(fqns), "lr": 1e-4}],
    }
    out = normalize_optimizer_state(module, opt, payload)
    assert list(out["state"]) == [fqns[0]]
    assert out["param_groups"][0]["params"] == list(fqns)


def test_fsdp_apply_remaps_before_set_optimizer_state_dict(monkeypatch) -> None:
    module, opt = _tiny()
    fqns = optimizer_param_fqns(module, opt)
    received: dict[str, object] = {}

    def fake_set_model(_module, model_state_dict=None, options=None):
        received["model"] = True

    def fake_set_opt(_module, _optimizer, optim_state_dict=None, options=None):
        received["state_keys"] = list(optim_state_dict["state"])
        received["params"] = list(optim_state_dict["param_groups"][0]["params"])

    monkeypatch.setattr("minakanushi.training.parallel.is_fsdp_wrapped", lambda _m: True)
    import torch.distributed.checkpoint.state_dict as sdict

    monkeypatch.setattr(sdict, "StateDictOptions", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(sdict, "set_model_state_dict", fake_set_model)
    monkeypatch.setattr(sdict, "set_optimizer_state_dict", fake_set_opt)

    payload = {
        "system": module.state_dict(),
        "optimizer": {
            "state": {0: {"step": torch.tensor(64), "exp_avg": torch.zeros_like(next(module.parameters()))}},
            "param_groups": [{"lr": 1e-3, "params": list(range(len(fqns)))}],
        },
    }
    apply_full_checkpoint(module, opt, payload)
    assert received["model"] is True
    assert received["state_keys"] == [fqns[0]]
    assert 0 not in received["state_keys"]
    assert received["params"] == list(fqns)
