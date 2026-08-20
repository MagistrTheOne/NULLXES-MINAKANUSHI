"""6.8B profile is a YAML contract. Do not construct MinakanushiSystem from it in CI."""

from __future__ import annotations

from pathlib import Path

from minakanushi.architecture.config import load_architecture
from minakanushi.training.parameter_inventory import estimate_parameters

ROOT = Path(__file__).resolve().parents[2]


def test_6_8b_config_dims_and_formula_without_building_the_net() -> None:
    path = ROOT / "configs" / "architecture" / "minakanushi_6_8b.yaml"
    model_copy = ROOT / "models" / "MINA-6.8B" / "architecture.yaml"
    cfg = load_architecture(path)
    pack = load_architecture(model_copy)
    cfg.validate()
    pack.validate()
    assert cfg.latent_dim == pack.latent_dim == 4096
    assert cfg.core_depth == 32
    assert cfg.world_slots == 512
    assert cfg.memory_slots == 1024
    assert cfg.memory_dim == cfg.latent_dim
    n = estimate_parameters(cfg)["total_estimate"]
    assert 6_700_000_000 < n < 6_900_000_000
    assert n == 6_799_130_646
