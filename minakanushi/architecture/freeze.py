"""Architecture freeze after 7aba976. Scale this MINA. Do not invent another."""

from __future__ import annotations

from minakanushi.architecture.config import ArchitectureConfig

FROZEN_AT = "7aba976"
FROZEN_PROFILE = "minakanushi_6_8b"
FROZEN_LATENT_DIM = 4096
FROZEN_STATE_DIM = 4096
FROZEN_MEMORY_DIM = 4096
FROZEN_WORLD_SLOTS = 512
FROZEN_MEMORY_SLOTS = 1024
FROZEN_CORE_DEPTH = 32
FROZEN_COGNITION_BUDGET = 4
FROZEN_PARAM_ESTIMATE = 6_799_130_646

ALLOWED_6_8B_GPU = ("H200", "B200", "B300")


def is_6_8b_profile(config: ArchitectureConfig) -> bool:
    return (
        int(config.latent_dim) == FROZEN_LATENT_DIM
        and int(config.core_depth) == FROZEN_CORE_DEPTH
        and int(config.world_slots) == FROZEN_WORLD_SLOTS
        and int(config.memory_slots) == FROZEN_MEMORY_SLOTS
    )


def assert_6_8b_frozen(config: ArchitectureConfig) -> None:
    if not is_6_8b_profile(config):
        raise ValueError("not the frozen minakanushi_6_8b profile")
    expected = {
        "latent_dim": FROZEN_LATENT_DIM,
        "state_dim": FROZEN_STATE_DIM,
        "memory_dim": FROZEN_MEMORY_DIM,
        "world_slots": FROZEN_WORLD_SLOTS,
        "memory_slots": FROZEN_MEMORY_SLOTS,
        "core_depth": FROZEN_CORE_DEPTH,
        "cognition.budget": FROZEN_COGNITION_BUDGET,
    }
    actual = {
        "latent_dim": int(config.latent_dim),
        "state_dim": int(config.state_dim),
        "memory_dim": int(config.memory_dim),
        "world_slots": int(config.world_slots),
        "memory_slots": int(config.memory_slots),
        "core_depth": int(config.core_depth),
        "cognition.budget": int(config.cognition.budget),
    }
    for key, want in expected.items():
        if actual[key] != want:
            raise ValueError(f"freeze violation {key}={actual[key]} (frozen {want} at {FROZEN_AT})")


def assert_may_construct(
    config: ArchitectureConfig,
    *,
    device: str,
    gpu_name: str = "",
    world_size: int = 1,
) -> None:
    """Refuse 6.8B construct on CPU, 6000 BW, or 1× H100 train."""
    if not is_6_8b_profile(config):
        return
    assert_6_8b_frozen(config)
    device_s = str(device).lower()
    if device_s == "cpu" or device_s.startswith("cpu"):
        raise RuntimeError(
            f"refusing to construct {FROZEN_PROFILE} on CPU (freeze {FROZEN_AT})"
        )
    name = str(gpu_name).upper().strip()
    if not name:
        raise RuntimeError(
            f"refusing to construct {FROZEN_PROFILE}: GPU name required (H200/B200/B300)"
        )
    if any(tag in name for tag in ("6000", "RTX PRO")):
        raise RuntimeError(f"refusing to construct {FROZEN_PROFILE} on {gpu_name}")
    if "H100" in name and int(world_size) < 2:
        raise RuntimeError("1× H100 80GB train is forbidden for 6.8B")
    if not any(tag in name for tag in ALLOWED_6_8B_GPU):
        raise RuntimeError(
            f"{FROZEN_PROFILE} train requires H200/B200/B300, got {gpu_name!r}"
        )
