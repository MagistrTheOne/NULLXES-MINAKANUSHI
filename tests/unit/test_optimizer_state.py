"""Integer-id Adam resume. Do not rewrite ids to dotted FQNs for FSDP2."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from minakanushi.training.optimizer_state import apply_optimizer_checkpoint, integer_optimizer_state
from minakanushi.training.parallel import apply_full_checkpoint


def _tiny() -> tuple[nn.Module, torch.optim.AdamW]:
    module = nn.Sequential(nn.Linear(4, 4, bias=True), nn.Linear(4, 2, bias=True))
    opt = torch.optim.AdamW(module.parameters(), lr=1e-3, foreach=False)
    loss = module(torch.ones(3, 4)).sum()
    loss.backward()
    opt.step()
    return module, opt


def test_as_local_tensor_uses_to_local_not_full_tensor() -> None:
    from minakanushi.training.optimizer_state import _as_local_tensor

    class DTensor:
        def __init__(self) -> None:
            self._local = torch.ones(2, 3)
            self.full_tensor_calls = 0

        def to_local(self) -> torch.Tensor:
            return self._local

        def full_tensor(self) -> torch.Tensor:
            self.full_tensor_calls += 1
            raise AssertionError("full_tensor all-gathers and dies on CPU DTensor")

        def detach(self):
            return self

    fake = DTensor()
    out = _as_local_tensor(fake)  # type: ignore[arg-type]
    assert torch.equal(out, torch.ones(2, 3))
    assert fake.full_tensor_calls == 0


def test_integer_nested_state_restores_moments() -> None:
    module, opt = _tiny()
    params = list(module.parameters())
    marker = torch.full_like(params[0], 3.5)
    payload = {
        "state": {
            0: {"step": torch.tensor(64), "exp_avg": marker, "exp_avg_sq": torch.zeros_like(params[0])},
        },
        "param_groups": [{"lr": 0.0001, "params": list(range(len(params)))}],
    }
    fresh = torch.optim.AdamW(module.parameters(), lr=1e-3, foreach=False)
    apply_optimizer_checkpoint(fresh, payload)
    first = list(fresh.param_groups[0]["params"])[0]
    assert torch.equal(fresh.state[first]["exp_avg"], marker)
    assert int(fresh.state[first]["step"].item()) == 64
    assert fresh.param_groups[0]["lr"] == 0.0001


def test_flat_state_28_step_fails_loud_then_loads_valid_ids() -> None:
    module, opt = _tiny()
    params = list(module.parameters())
    flat: dict[str, object] = {
        "param_groups.0.lr": 1e-3,
        "param_groups.0.weight_decay": 0.01,
    }
    for i, param in enumerate(params):
        flat[f"param_groups.0.params.{i}"] = i
        flat[f"state.{i}.step"] = torch.tensor(64)
        flat[f"state.{i}.exp_avg"] = torch.zeros_like(param)
        flat[f"state.{i}.exp_avg_sq"] = torch.zeros_like(param)
    flat["state.28.step"] = torch.tensor(64)
    try:
        apply_optimizer_checkpoint(opt, flat)
        raise AssertionError("index 28 must fail loud on a 4-parameter net")
    except ValueError as exc:
        assert "28" in str(exc)
    del flat["state.28.step"]
    nested = integer_optimizer_state(flat)
    assert 0 in nested["state"] or "0" in {str(k) for k in nested["state"]}
    apply_optimizer_checkpoint(opt, flat)
    first = list(opt.param_groups[0]["params"])[0]
    assert int(opt.state[first]["step"].item()) == 64


def test_fqn_optimizer_state_is_rejected() -> None:
    module, opt = _tiny()
    payload = {
        "state": {"perception.vector.net.0.weight": {"step": torch.tensor(7)}},
        "param_groups": [{"params": [0], "lr": 1e-4}],
    }
    try:
        apply_optimizer_checkpoint(opt, payload)
        raise AssertionError("FQN optimizer keys must fail loud")
    except ValueError as exc:
        assert "FQN-keyed" in str(exc)


def test_fsdp_apply_loads_integer_optimizer_not_set_optimizer_state_dict(monkeypatch) -> None:
    module, opt = _tiny()
    received: dict[str, object] = {}

    def fake_set_model(_module, model_state_dict=None, options=None):
        received["model"] = True

    def fake_set_opt(*_args, **_kwargs):
        raise AssertionError("set_optimizer_state_dict must not run for mixed FSDP2 resume")

    monkeypatch.setattr("minakanushi.training.parallel.is_fsdp_wrapped", lambda _m: True)
    import torch.distributed.checkpoint.state_dict as sdict

    monkeypatch.setattr(sdict, "StateDictOptions", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(sdict, "set_model_state_dict", fake_set_model)
    monkeypatch.setattr(sdict, "set_optimizer_state_dict", fake_set_opt)

    marker = torch.full_like(next(module.parameters()), 2.25)
    payload = {
        "system": module.state_dict(),
        "optimizer": {
            "state": {0: {"step": torch.tensor(64), "exp_avg": marker, "exp_avg_sq": torch.zeros_like(marker)}},
            "param_groups": [{"lr": 1e-3, "params": [0]}],
        },
    }
    apply_full_checkpoint(module, opt, payload)
    assert received["model"] is True
    first = list(opt.param_groups[0]["params"])[0]
    assert torch.equal(opt.state[first]["exp_avg"], marker)
