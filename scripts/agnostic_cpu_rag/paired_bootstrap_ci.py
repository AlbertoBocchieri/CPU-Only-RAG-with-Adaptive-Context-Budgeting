#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_rows(path: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[str(row["qid"])] = row
    return rows


def nested_get(record: dict[str, Any], path: str) -> Any:
    node: Any = record
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    if low == high:
        return float(sorted_values[low])
    frac = pos - low
    return float(sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paired bootstrap CI for a metric delta.")
    parser.add_argument("base_per_query_jsonl")
    parser.add_argument("new_per_query_jsonl")
    parser.add_argument("--metric-path", default="metrics.f1")
    parser.add_argument("--base-metric-path", default=None)
    parser.add_argument("--new-metric-path", default=None)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_rows = load_rows(args.base_per_query_jsonl)
    new_rows = load_rows(args.new_per_query_jsonl)
    qids = [qid for qid in base_rows if qid in new_rows]
    base_metric_path = str(args.base_metric_path or args.metric_path)
    new_metric_path = str(args.new_metric_path or args.metric_path)
    deltas: list[float] = []
    better = worse = ties = 0

    for qid in qids:
        base_value = nested_get(base_rows[qid], base_metric_path)
        new_value = nested_get(new_rows[qid], new_metric_path)
        if base_value is None or new_value is None:
            continue
        delta = float(new_value) - float(base_value)
        deltas.append(delta)
        better += delta > 0
        worse += delta < 0
        ties += delta == 0

    rng = random.Random(args.seed)
    boots: list[float] = []
    if deltas:
        for _ in range(int(args.resamples)):
            sample = [deltas[rng.randrange(len(deltas))] for _ in range(len(deltas))]
            boots.append(sum(sample) / len(sample))
        boots.sort()
    mean_delta = (sum(deltas) / len(deltas)) if deltas else 0.0

    out = {
        "base_metric_path": str(base_metric_path),
        "new_metric_path": str(new_metric_path),
        "num_qids": len(deltas),
        "mean_delta": float(mean_delta),
        "ci95_low": float(quantile(boots, 0.025)) if boots else 0.0,
        "ci95_high": float(quantile(boots, 0.975)) if boots else 0.0,
        "better": int(better),
        "worse": int(worse),
        "tie": int(ties),
    }
    print(json.dumps(out, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
