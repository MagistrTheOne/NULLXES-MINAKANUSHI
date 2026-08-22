"""Optional Hugging Face AutoConfig / AutoModel registration.

MINA is not CausalLM. This does not construct minakanushi_6_8b.
Runtime load remains load_mina on a GPU-class machine.

    python scripts/register_hf_architecture.py
"""

from __future__ import annotations

from minakanushi.architecture.config import ArchitectureConfig


HUB_MODEL_TYPE = "minakanushi"
RESEARCH_LATENT = 4096

MinakanushiHFConfig: type | None = None
MinakanushiHubModel: type | None = None


def refuse_if_research_scale(config: object) -> None:
    """Hub class is a type tag. 6.8B is load_mina on H200/B300."""
    latent = int(getattr(config, "latent_dim", 0) or 0)
    if latent >= RESEARCH_LATENT:
        raise RuntimeError(
            "refusing to construct minakanushi_6_8b via AutoModel. "
            "Use load_mina on H200/B300. This Hub class is a type tag, not the runtime."
        )


def minakanushi_hf_config_dict(arch: ArchitectureConfig | None = None) -> dict:
    cfg = {
        "model_type": HUB_MODEL_TYPE,
        "architecture": "MINAKANUSHI",
        "organization": "NULLXES",
        "short_name": "MINA",
        "format": "safetensors_mirror",
        "runtime": "load_mina",
        "world_model": True,
        "action_output": "ActionIntent",
        "not_a_language_model": True,
        "auto_map": {
            "AutoConfig": "minakanushi.hub.MinakanushiHFConfig",
            "AutoModel": "minakanushi.hub.MinakanushiHubModel",
        },
    }
    if arch is not None:
        ident = arch.identity
        cfg.update(
            {
                "latent_dim": arch.latent_dim,
                "state_dim": arch.state_dim,
                "memory_dim": arch.memory_dim,
                "core_depth": arch.core_depth,
                "world_slots": arch.world_slots,
                "memory_slots": arch.memory_slots,
                "native_runtime": ident.native_runtime,
            }
        )
    return cfg


def _ensure_hf_classes() -> tuple[type, type]:
    global MinakanushiHFConfig, MinakanushiHubModel
    if MinakanushiHFConfig is not None and MinakanushiHubModel is not None:
        return MinakanushiHFConfig, MinakanushiHubModel
    try:
        from transformers import PretrainedConfig, PreTrainedModel
    except ImportError as exc:
        raise SystemExit("optional extra: pip install transformers") from exc

    class _MinakanushiHFConfig(PretrainedConfig):
        model_type = HUB_MODEL_TYPE

        def __init__(self, **kwargs):
            if str(kwargs.get("model_type") or HUB_MODEL_TYPE) in {"llama", "gpt2"}:
                raise ValueError("MINAKANUSHI must not register as a causal LM")
            super().__init__(**kwargs)
            self.architecture = kwargs.get("architecture", "MINAKANUSHI")
            self.runtime = kwargs.get("runtime", "load_mina")
            self.action_output = kwargs.get("action_output", "ActionIntent")
            self.world_model = bool(kwargs.get("world_model", True))
            self.not_a_language_model = True

    class _MinakanushiHubModel(PreTrainedModel):
        config_class = _MinakanushiHFConfig

        def __init__(self, config: _MinakanushiHFConfig):
            refuse_if_research_scale(config)
            super().__init__(config)
            import torch

            self.register_buffer("_mina_hub_tag", torch.zeros(1))

        def forward(self, *args, **kwargs):
            raise RuntimeError("MinakanushiHubModel has no token forward. Use MinakanushiRuntime.")

    MinakanushiHFConfig = _MinakanushiHFConfig
    MinakanushiHubModel = _MinakanushiHubModel
    return MinakanushiHFConfig, MinakanushiHubModel


def register_minakanushi() -> tuple[type, type]:
    """Register AutoConfig + AutoModel. Never AutoModelForCausalLM."""
    try:
        from transformers import AutoConfig, AutoModel
    except ImportError as exc:
        raise SystemExit("optional extra: pip install transformers") from exc

    config_cls, model_cls = _ensure_hf_classes()
    AutoConfig.register(HUB_MODEL_TYPE, config_cls)
    AutoModel.register(config_cls, model_cls)
    return config_cls, model_cls
