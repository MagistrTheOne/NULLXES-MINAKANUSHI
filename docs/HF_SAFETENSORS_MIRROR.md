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
python scripts/export_safetensors.py \
  --mina minakanushi_stage0_step128.mina \
  --out MINAKANUSHI-6.8B
```

`--cards-only` writes `config.json`, `minakanushi_config.json`,
`MINAKANUSHI_CARD.json`, `minakanushi_runtime.json`, `LICENSE`, and
`generation/NO` without touching 27 GB.

Weights go out as bf16 shards (~13.6 GB for 6.8B), 5 GiB each, plus
`model.safetensors.index.json`. Optimizer and RNG stay inside `.mina`.

Forbidden in the mirror card:

```text
model_type: llama
architectures: [LlamaForCausalLM]
chat_template
generation_config.json
```

Load path remains `load_mina(...)`. `AutoModel.from_pretrained` is a type tag
(`AutoConfig` / `AutoModel`, never `AutoModelForCausalLM`) and refuses
`latent_dim >= 4096` construct. It is not the runtime.

Every published checkpoint after v0.3.1 ships **both**:

```text
*.mina                 canonical runtime / resume / identity / optimizer
safetensors shards     Hub weight mirror (bf16)
```

Do not convert step64. Do not construct 6.8B to export. Export on H200/B300,
then `scripts/test_hf_reload.py` must PASS (0-dim scalars are valid if they
match the shard header). Upload is `scripts/publish_v031_hf.py`.

v0.3.1 Hub card is a **research checkpoint, not accepted**.
`compare_v031.py` is the verdict, not train loss.
