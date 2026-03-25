#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def round_to_multiple(value: float, multiple: int) -> int:
    return int(round(float(value) / float(multiple)) * multiple)


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive a fixed-cap override from an EWMA runtime report.")
    parser.add_argument("runtime_report_json")
    parser.add_argument("--system", choices=["agnostic", "legacy"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--multiple", type=int, default=32)
    args = parser.parse_args()

    report = json.loads(Path(args.runtime_report_json).read_text(encoding="utf-8"))
    stats = (((report.get("hardware_adaptive") or {}).get("post_warmup_budget_cap_tokens")) or {})
    mean_cap = stats.get("mean")
    if mean_cap is None:
        raise ValueError("post_warmup_budget_cap_tokens.mean missing in runtime report")
    fixed_cap = round_to_multiple(float(mean_cap), int(args.multiple))

    if args.system == "agnostic":
        payload = {"runtime": {"fixed_cap_tokens": int(fixed_cap)}}
    else:
        payload = {"context_budgeting": {"fixed_cap_tokens": int(fixed_cap)}}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(json.dumps({"fixed_cap_tokens": int(fixed_cap), "output": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
