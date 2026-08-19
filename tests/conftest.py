from pathlib import Path

from minakanushi.architecture.config import load_config
from minakanushi.runtime.engine import MinakanushiEngine


ROOT = Path(__file__).resolve().parents[2]


def cpu_config():
    return load_config(
        ROOT / "configs" / "architecture" / "cpu_dev.yaml",
        runtime_path=ROOT / "configs" / "runtime" / "cpu.yaml",
        simulation_path=ROOT / "configs" / "simulation" / "milestone1.yaml",
    )


def build_engine() -> MinakanushiEngine:
    return MinakanushiEngine(cpu_config())
