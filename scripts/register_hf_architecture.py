"""Register MINAKANUSHI with Hugging Face AutoConfig / AutoModel. Not CausalLM."""

from __future__ import annotations

import json

from minakanushi.hub import HUB_MODEL_TYPE, minakanushi_hf_config_dict, register_minakanushi


def main() -> None:
    config_cls, model_cls = register_minakanushi()
    print(
        json.dumps(
            {
                "model_type": HUB_MODEL_TYPE,
                "config_class": f"{config_cls.__module__}.{config_cls.__name__}",
                "model_class": f"{model_cls.__module__}.{model_cls.__name__}",
                "auto_model_for_causal_lm": False,
                "runtime": "load_mina",
                "config": minakanushi_hf_config_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
