"""Structured runtime telemetry."""

from __future__ import annotations

import json
import logging
from time import perf_counter

from minakanushi.architecture.outputs import CycleTelemetry


class TelemetryLogger:
    def __init__(self, level: str = "INFO") -> None:
        self.log = logging.getLogger("nullxes.minakanushi")
        if not self.log.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self.log.addHandler(handler)
        self.log.setLevel(level)

    def emit(self, telemetry: CycleTelemetry) -> None:
        payload = {
            "cycle_id": telemetry.cycle_id,
            "physical_time": telemetry.physical_time,
            "observation_count": telemetry.observation_count,
            "entity_count": telemetry.entity_count,
            "event_count": telemetry.event_count,
            "world_state_confidence": telemetry.world_state_confidence,
            "uncertainty": telemetry.uncertainty,
            "memory_reads": telemetry.memory_reads,
            "memory_writes": telemetry.memory_writes,
            "future_branches": telemetry.future_branches,
            "candidate_strategies": telemetry.candidate_strategies,
            "rejected_strategies": telemetry.rejected_strategies,
            "rejection_reasons": list(telemetry.rejection_reasons),
            "selected_strategy": telemetry.selected_strategy,
            "cognition_cycles": telemetry.cognition_cycles,
            "latency_ms": telemetry.latency_ms,
        }
        self.log.info(json.dumps(payload, ensure_ascii=True))


class LatencyClock:
    def __init__(self) -> None:
        self.mark = perf_counter()

    def ms(self) -> float:
        return (perf_counter() - self.mark) * 1000.0
