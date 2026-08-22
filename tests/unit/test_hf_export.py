"""Safetensors is a Hugging Face mirror. Canonical runtime stays *.mina."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
import torch
import yaml

from minakanushi.training.hf_export import (
    assert_not_llm_card,
    export_hf_mirror,
    hf_config,
)


def _manifest() -> dict:
    return {
        "architecture": "MINAKANUSHI",
        "organization": "NULLXES",
        "system_class": "adaptive_situational_intelligence",
        "generation": 1,
        "architecture_version": "0.1",
        "native_runtime": "nullxes",
        "latent_dim": 8,
        "state_dim": 8,
        "memory_dim": 8,
        "world_slots": 4,
        "memory_slots": 8,
        "core_depth": 2,
        "uncertainty_channels": 2,
        "future_branches": 3,
    }


def _write_mina(path: Path, *, sharded: bool) -> None:
    system = {
        "weight": torch.randn(16, 32),
        "bias": torch.randn(16),
    }
    optimizer = {"exp_avg": torch.ones(8)}
    payload = {"system": system, "optimizer": optimizer, "runtime": None}
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.yaml", yaml.safe_dump(_manifest(), sort_keys=False))
        zf.writestr("architecture.yaml", "profile_name: cpu_probe\n")
        if sharded:
            buf = io.BytesIO()
            torch.save({"weight": system["weight"]}, buf)
            zf.writestr("weights/system-00000.pt", buf.getvalue())
            buf = io.BytesIO()
            torch.save({"bias": system["bias"]}, buf)
            zf.writestr("weights/system-00001.pt", buf.getvalue())
            buf = io.BytesIO()
            torch.save({"optimizer": optimizer, "runtime": None}, buf)
            zf.writestr("weights/sidecar.pt", buf.getvalue())
        else:
            buf = io.BytesIO()
            torch.save(payload, buf)
            zf.writestr("weights.pt", buf.getvalue())


def test_hf_config_is_not_llama() -> None:
    config = hf_config(_manifest(), canonical_checkpoint="probe.mina")
    assert config["model_type"] == "minakanushi"
    assert config["runtime"] == "load_mina"
    assert config["action_output"] == "ActionIntent"
    assert config["auto_map"]["AutoConfig"] == "minakanushi.hub.MinakanushiHFConfig"
    assert "architectures" not in config
    with pytest.raises(ValueError, match="llama"):
        assert_not_llm_card({"model_type": "llama"})
    with pytest.raises(ValueError, match="Transformers"):
        assert_not_llm_card({"architectures": ["LlamaForCausalLM"]})


def test_export_mirror_drops_optimizer_and_writes_bf16_shards(tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file

    mina = tmp_path / "probe_step3.mina"
    _write_mina(mina, sharded=False)
    out = tmp_path / "mirror"
    result = export_hf_mirror(mina, out, shard_bytes=64)
    index = json.loads((out / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert result["parameters"] == 32 * 16 + 16
    assert "exp_avg" not in index["weight_map"]
    assert set(index["weight_map"]) == {"weight", "bias"}
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    card = json.loads((out / "MINAKANUSHI_CARD.json").read_text(encoding="utf-8"))
    runtime = json.loads((out / "minakanushi_runtime.json").read_text(encoding="utf-8"))
    assert config["model_type"] == "minakanushi"
    assert config["format"] == "safetensors_mirror"
    assert card["not_a_chat_model"] is True
    assert runtime["transformers_auto_model"] is False
    assert runtime["transformers_hub_role"] == "type_tag_only"
    assert (out / "minakanushi_config.json").is_file()
    assert (out / "generation" / "NO").is_file()
    restored: dict[str, torch.Tensor] = {}
    for name in result["shards"]:
        restored.update(load_file(str(out / name)))
    assert restored["weight"].dtype == torch.bfloat16
    assert restored["bias"].dtype == torch.bfloat16
    blob = (out / "config.json").read_text(encoding="utf-8")
    assert "LlamaForCausalLM" not in blob
    assert "llama" not in blob.lower()


def test_export_sharded_mina_and_cards_only(tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    mina = tmp_path / "probe_step4.mina"
    _write_mina(mina, sharded=True)
    out = tmp_path / "mirror"
    result = export_hf_mirror(mina, out, shard_bytes=10_000_000)
    assert result["parameters"] == 32 * 16 + 16
    cards = tmp_path / "cards"
    license_src = tmp_path / "LICENSE"
    license_src.write_text("NULLXES MINAKANUSHI\n", encoding="utf-8")
    card_result = export_hf_mirror(mina, cards, cards_only=True, license_path=license_src)
    assert card_result["cards_only"] is True
    assert not list(cards.glob("*.safetensors"))
    assert (cards / "config.json").exists()
    assert (cards / "minakanushi_config.json").exists()
    assert (cards / "LICENSE").exists()
    assert (cards / "architecture.yaml").exists()


def test_export_roundtrip_reloads_bf16(tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    from minakanushi.training.export_roundtrip import export_and_reload

    mina = tmp_path / "probe_step128.mina"
    _write_mina(mina, sharded=False)
    out = tmp_path / "MINAKANUSHI-6.8B"
    result = export_and_reload(mina, out, shard_bytes=64)
    assert result["reloaded"] is True
    assert result["n_tensors"] == 2
    assert (out / "minakanushi_config.json").is_file()
