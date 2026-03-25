#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def nested_get(node: dict[str, Any], *parts: str) -> Any:
    value: Any = node
    for part in parts:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def metric(report: dict[str, Any], *parts: str) -> float | None:
    value = nested_get(report, *parts)
    return None if value is None else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare EWMA vs fixed-cap tracking reports.")
    parser.add_argument("ewma_report_json")
    parser.add_argument("fixed_report_json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    ewma = load(args.ewma_report_json)
    fixed = load(args.fixed_report_json)

    ewma_error = metric(ewma, "prefill_tracking", "abs_prefill_target_error_ms_post_warmup", "mean")
    fixed_error = metric(fixed, "prefill_tracking", "abs_prefill_target_error_ms_post_warmup", "mean")
    ewma_overshoot = metric(ewma, "prefill_tracking", "overshoot_rate_post_warmup")
    fixed_overshoot = metric(fixed, "prefill_tracking", "overshoot_rate_post_warmup")
    ewma_cap_std = metric(ewma, "hardware_adaptive", "budget_cap_tokens", "std")
    fixed_cap_std = metric(fixed, "hardware_adaptive", "budget_cap_tokens", "std")

    out = {
        "ewma": {
            "f1": metric(ewma, "summary_metrics", "f1"),
            "em": metric(ewma, "summary_metrics", "em"),
            "abs_prefill_target_error_ms_post_warmup_mean": ewma_error,
            "overshoot_rate_post_warmup": ewma_overshoot,
            "budget_cap_tokens_std": ewma_cap_std,
        },
        "fixed": {
            "f1": metric(fixed, "summary_metrics", "f1"),
            "em": metric(fixed, "summary_metrics", "em"),
            "abs_prefill_target_error_ms_post_warmup_mean": fixed_error,
            "overshoot_rate_post_warmup": fixed_overshoot,
            "budget_cap_tokens_std": fixed_cap_std,
        },
        "gates": {
            "error_lower_is_better": (ewma_error is not None and fixed_error is not None and ewma_error < fixed_error),
            "overshoot_lower_is_better": (
                ewma_overshoot is not None and fixed_overshoot is not None and ewma_overshoot < fixed_overshoot
            ),
            "cap_std_lower_or_equal": (ewma_cap_std is not None and fixed_cap_std is not None and ewma_cap_std <= fixed_cap_std),
        },
    }

    payload = json.dumps(out, indent=2, ensure_ascii=True)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
