"""CPU contract gate for v0.3.1. Not training. Does not construct 6.8B.

FAIL must be FAIL. PASS must be PASS. Expected before H200: Gate C/E fail.

    python scripts/gate_v031_acceptance.py --dataset dataset/mina_6_8b_v03 --split heldout
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minakanushi.training.capability import cpu_trainer, gate_c_causality, gate_c_is_honest, gate_e_is_honest, gate_e_memory
from minakanushi.training.curriculum_audit import audit_curriculum
from minakanushi.training.heldout import write_heldout_split

ROOT = Path(__file__).resolve().parents[1]


def _honest_c(row: dict) -> bool:
    return gate_c_is_honest(row)


def _honest_e(row: dict) -> bool:
    return gate_e_is_honest(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset" / "mina_6_8b_v03")
    parser.add_argument("--split", type=str, default="heldout")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "v031" / "acceptance.json")
    args = parser.parse_args()
    if not (args.dataset / args.split / "index.jsonl").is_file():
        write_heldout_split(args.dataset)
    audit = audit_curriculum(args.dataset, gate=False, split=args.split)
    trainer = cpu_trainer(12)
    c = gate_c_causality(trainer)
    e = gate_e_memory(trainer, length=32)
    contract = {
        "C_pass_matches_revision": _honest_c(c),
        "E_pass_matches_ade": _honest_e(e),
        "heldout_exists": audit["n_episodes"] > 0,
        "pwm_false": audit["pwm"] is False,
    }
    report = {
        "gate": "v0.3.1_acceptance",
        "split": args.split,
        "n_episodes": audit["n_episodes"],
        "pwm": audit["pwm"],
        "C": {
            "pass": c["pass"],
            "revision_detected": c["revision_detected"],
            "picture_in_picture_out": c["picture_in_picture_out"],
            "honest": contract["C_pass_matches_revision"],
        },
        "E": {
            "pass": e["pass"],
            "memory_ade_on": e["memory_ade_on"],
            "memory_ade_off": e["memory_ade_off"],
            "memory_helps_future": e["memory_helps_future"],
            "honest": contract["E_pass_matches_ade"],
        },
        "not_a_claim": "C/E FAIL before H200 is the honest baseline, not a broken script.",
        "contract": contract,
        "pass": all(contract.values()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit("v0.3.1 acceptance contract broken (pass flags dishonest or pack missing)")


if __name__ == "__main__":
    main()
