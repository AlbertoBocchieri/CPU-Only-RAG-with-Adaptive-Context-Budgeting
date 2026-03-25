#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize comparison guard activation patterns.")
    parser.add_argument("per_query_jsonl")
    parser.add_argument("--max-examples-per-pattern", type=int, default=3)
    args = parser.parse_args()

    rows = load_rows(args.per_query_jsonl)
    total = len(rows)
    triggered_rows = []
    cue_counter: Counter[str] = Counter()
    prefix_counter: Counter[str] = Counter()
    overlap_counter: Counter[str] = Counter()
    examples_by_pattern: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        controller = dict(row.get("context_controller", {}))
        if not controller.get("comparison_guard_triggered", False):
            continue
        triggered_rows.append(row)
        prefix = str(controller.get("comparison_guard_prefix", "")).strip().lower()
        if prefix:
            prefix_counter[prefix] += 1
        matched_cues = [str(cue) for cue in controller.get("comparison_guard_matched_cues", [])]
        for cue in matched_cues:
            cue_counter[cue] += 1
            if len(examples_by_pattern[cue]) < args.max_examples_per_pattern:
                examples_by_pattern[cue].append({"qid": row.get("qid", ""), "query": row.get("query", "")})
        if prefix and len(examples_by_pattern[f"prefix:{prefix}"]) < args.max_examples_per_pattern:
            examples_by_pattern[f"prefix:{prefix}"].append({"qid": row.get("qid", ""), "query": row.get("query", "")})
        overlap_key = " + ".join(sorted(set(matched_cues))) if matched_cues else "<no_cues>"
        overlap_counter[overlap_key] += 1

    triggered = len(triggered_rows)
    report = {
        "total_queries": total,
        "triggered_queries": triggered,
        "activation_rate": (float(triggered) / float(total)) if total else 0.0,
        "prefix_counts": dict(prefix_counter.most_common()),
        "cue_counts": dict(cue_counter.most_common()),
        "overlap_counts": dict(overlap_counter.most_common()),
        "top_examples_by_pattern": dict(examples_by_pattern),
    }
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
