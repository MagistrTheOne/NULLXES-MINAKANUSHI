"""No architecture drift. Does not construct 6.8B.

    python scripts/check_freeze.py
    python scripts/check_freeze.py --checkpoint minakanushi_stage0_step128.mina
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minakanushi.training.freeze_check import check_mina_freeze, check_yaml_freeze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None, help="*.mina; omit to check frozen YAML only")
    args = parser.parse_args()
    yaml_report = check_yaml_freeze()
    report = {"yaml": yaml_report, "checkpoint": None, "pass": yaml_report["pass"]}
    if args.checkpoint is not None:
        mina_report = check_mina_freeze(args.checkpoint)
        report["checkpoint"] = mina_report
        report["pass"] = yaml_report["pass"] and mina_report["pass"]
        report["dwc_contract_match"] = yaml_report["contract_hash"] == mina_report["contract_hash"]
        if not report["dwc_contract_match"]:
            report["pass"] = False
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit("architecture drift")


if __name__ == "__main__":
    main()
