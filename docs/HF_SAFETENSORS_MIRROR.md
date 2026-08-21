# Hugging Face safetensors mirror

Canonical artifact is `*.mina`. Safetensors is a public mirror for Hub discovery
(parameter badge, download stats). It is not the runtime and not a Transformers
checkpoint.

```text
MINA checkpoint
        |
        +---- *.mina                 native runtime, resume, identity
        |
        +---- safetensors shards     HF vitirine only
```

Do **not** convert v0.1 `step64.mina`. That file is an engineering witness
(construct / forward / backward / AdamW / save / load). Mirror after the next
full B300 segment that has passed IdentityBound → JSON curriculum → train →
Acceptance Gate.

```text
python scripts/export_hf.py \
  --mina path/to/final.mina \
  --out /workspace/hf_mirror_v02
```

`--cards-only` writes `config.json`, `MINAKANUSHI_CARD.json`,
`minakanushi_runtime.json`, and `generation/NO` without touching 26 GB.

Weights go out as bf16 shards (~13.6 GB for 6.8B), 5 GiB each, plus
`model.safetensors.index.json`. Optimizer and RNG stay inside `.mina`.

Forbidden in the mirror card:

```text
model_type: llama
architectures: [LlamaForCausalLM]
chat_template
generation_config.json
```

Load path remains `load_mina(...)`. `AutoModel.from_pretrained` is not supported.
Upload is a later step. This document does not publish weights.
