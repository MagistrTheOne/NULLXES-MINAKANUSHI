"""Print dataset balance. Not a training run."""

from __future__ import annotations

import argparse
from pathlib import Path

from simulations.synthetic_world.balance import load_records, tally_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = tally_records(load_records(args.root))
    print("episodes", report.n_episodes)
    print("scenarios", report.scenario_count)
    print("actions", report.action_count)
    print("events", report.event_count)
    print("occlusions", report.occlusion_count)
    print("corrections", report.correction_count)
    print("conflicts", report.conflict_count)
    print("max_scenario_fraction", round(report.max_scenario_fraction(), 3))


if __name__ == "__main__":
    main()
