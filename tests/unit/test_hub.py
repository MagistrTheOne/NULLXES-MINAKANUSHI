"""Hub registration is a type tag. Not CausalLM. Does not construct 6.8B."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from minakanushi.hub import HUB_MODEL_TYPE, minakanushi_hf_config_dict, refuse_if_research_scale


def test_hub_config_is_not_causal_lm() -> None:
    cfg = minakanushi_hf_config_dict()
    assert cfg["model_type"] == "minakanushi"
    assert cfg["runtime"] == "load_mina"
    assert cfg["action_output"] == "ActionIntent"
    assert cfg["auto_map"]["AutoModel"] == "minakanushi.hub.MinakanushiHubModel"
    assert "LlamaForCausalLM" not in str(cfg)
    assert "AutoModelForCausalLM" not in str(cfg)


def test_research_scale_is_refused_without_constructing() -> None:
    with pytest.raises(RuntimeError, match="load_mina"):
        refuse_if_research_scale(SimpleNamespace(latent_dim=4096))
    refuse_if_research_scale(SimpleNamespace(latent_dim=8))


def test_register_is_not_causal_lm() -> None:
    pytest.importorskip("transformers")
    from minakanushi.hub import register_minakanushi

    config_cls, model_cls = register_minakanushi()
    cfg = config_cls(model_type=HUB_MODEL_TYPE, latent_dim=4096)
    with pytest.raises(RuntimeError, match="load_mina"):
        refuse_if_research_scale(cfg)
    assert "CausalLM" not in model_cls.__name__
    assert config_cls.model_type == "minakanushi"
