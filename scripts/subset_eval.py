#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag_cpu.metrics import qa_scores, summarize_list
from rag_cpu.utils import save_json


def _load_qids(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if path.suffix.lower() == ".json":
        payload = json.loads(raw)
        if not isinstance(payload, list):
            raise ValueError(f"qids file must be a JSON list: {path}")
        return [str(x) for x in payload if str(x).strip()]
    return [line.strip() for line in raw.splitlines() if line.strip()]


def _safe_float(value: Any) -> float | None:
    try:
        f = float(value)
    except Exception:
        return None
    if f != f:
        return None
    return f


def _extract_em_f1(row: dict[str, Any]) -> tuple[float, float]:
    failed = bool(row.get("failed", False))
    metrics = row.get("answer_metrics_per_query", {})
    if not isinstance(metrics, dict):
        metrics = {}

    em = _safe_float(metrics.get("em", None))
    f1 = _safe_float(metrics.get("f1", None))
    if em is not None and f1 is not None:
        return float(em), float(f1)

    # Legacy fallback where both might be encoded in a single key.
    emf1 = metrics.get("em/f1", None)
    if emf1 is not None:
        if isinstance(emf1, str) and "/" in emf1:
            left, right = emf1.split("/", 1)
            em2 = _safe_float(left.strip())
            f12 = _safe_float(right.strip())
            if em2 is not None and f12 is not None:
                return float(em2), float(f12)
        if isinstance(emf1, (list, tuple)) and len(emf1) >= 2:
            em2 = _safe_float(emf1[0])
            f12 = _safe_float(emf1[1])
            if em2 is not None and f12 is not None:
                return float(em2), float(f12)

    if failed:
        return 0.0, 0.0

    prediction = str(row.get("prediction", ""))
    refs = row.get("answer_refs", [])
    if not isinstance(refs, list):
        refs = []
    qas = qa_scores(prediction, refs)
    return float(qas.get("em", 0.0)), float(qas.get("f1", 0.0))


def _read_per_query(path: Path, qid_set: set[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "qid" not in row:
                raise ValueError(f"missing 'qid' in per_query row at line {idx} ({path})")
            qid = str(row["qid"])
            if qid not in qid_set:
                continue
            if qid in out:
                # Keep deterministic first occurrence.
                continue
            out[qid] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fixed subset from per_query.jsonl")
    parser.add_argument("--per-query", required=True)
    parser.add_argument("--qids", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    per_query_path = Path(args.per_query)
    qids_path = Path(args.qids)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qids = _load_qids(qids_path)
    qids = [str(x) for x in qids]
    qid_set = set(qids)
    if len(qids) != len(qid_set):
        raise ValueError("qids input contains duplicates; expected canonical unique qids")

    rows_by_qid = _read_per_query(per_query_path, qid_set)
    matched = [qid for qid in qids if qid in rows_by_qid]
    missing = [qid for qid in qids if qid not in rows_by_qid]

    if len(matched) != 1000:
        raise RuntimeError(
            f"hard fail: matched_qids={len(matched)} (expected 1000). missing={len(missing)}"
        )

    em_values: list[float] = []
    f1_values: list[float] = []
    ttft_values: list[float] = []
    t_total_values: list[float] = []
    ctx_values: list[float] = []
    out_values: list[float] = []
    failed_count = 0

    for qid in matched:
        row = rows_by_qid[qid]
        em, f1 = _extract_em_f1(row)
        em_values.append(float(em))
        f1_values.append(float(f1))

        latency = row.get("latency_ms", {}) if isinstance(row.get("latency_ms", {}), dict) else {}
        tokens = row.get("tokens", {}) if isinstance(row.get("tokens", {}), dict) else {}

        ttft_values.append(float(_safe_float(latency.get("ttft_ms", 0.0)) or 0.0))
        t_total_values.append(float(_safe_float(latency.get("t_total_ms", 0.0)) or 0.0))
        ctx_values.append(float(_safe_float(tokens.get("context_tokens", 0.0)) or 0.0))
        out_values.append(float(_safe_float(tokens.get("output_tokens", 0.0)) or 0.0))

        if bool(row.get("failed", False)):
            failed_count += 1

    summary = {
        "task": "qa_subset_eval",
        "source_per_query": str(per_query_path),
        "qids_path": str(qids_path),
        "matched_qids_count": int(len(matched)),
        "num_queries": int(len(matched)),
        "generation": {
            "EM": float(sum(em_values) / len(em_values)) if em_values else 0.0,
            "F1": float(sum(f1_values) / len(f1_values)) if f1_values else 0.0,
            "failure_rate": float(failed_count / max(1, len(matched))),
            "em_values": em_values,
            "f1_values": f1_values,
        },
        "latency_ms": {
            "ttft_ms": summarize_list(ttft_values),
            "t_total_ms": summarize_list(t_total_values),
        },
        "tokens": {
            "context_tokens": summarize_list(ctx_values),
            "output_tokens": summarize_list(out_values),
        },
        "missing_qids": missing,
        "artifacts": {
            "subset_summary_json": str(out_dir / "subset_summary.json"),
            "subset_summary_md": str(out_dir / "subset_summary.md"),
        },
    }

    save_json(out_dir / "subset_summary.json", summary)

    lines = [
        "# Subset Summary",
        "",
        f"- source_per_query: `{per_query_path}`",
        f"- qids_path: `{qids_path}`",
        f"- matched_qids_count: {len(matched)}",
        "",
        "## Generation",
        "",
        f"- EM: {summary['generation']['EM']:.4f}",
        f"- F1: {summary['generation']['F1']:.4f}",
        f"- failure_rate: {summary['generation']['failure_rate']:.4f}",
        "",
        "## Latency",
        "",
        (
            f"- ttft_ms: mean={summary['latency_ms']['ttft_ms']['mean']:.3f} "
            f"p50={summary['latency_ms']['ttft_ms']['p50']:.3f} "
            f"p95={summary['latency_ms']['ttft_ms']['p95']:.3f}"
        ),
        (
            f"- t_total_ms: mean={summary['latency_ms']['t_total_ms']['mean']:.3f} "
            f"p50={summary['latency_ms']['t_total_ms']['p50']:.3f} "
            f"p95={summary['latency_ms']['t_total_ms']['p95']:.3f}"
        ),
        "",
        "## Tokens",
        "",
        (
            f"- context_tokens: mean={summary['tokens']['context_tokens']['mean']:.3f} "
            f"p50={summary['tokens']['context_tokens']['p50']:.3f} "
            f"p95={summary['tokens']['context_tokens']['p95']:.3f}"
        ),
        (
            f"- output_tokens: mean={summary['tokens']['output_tokens']['mean']:.3f} "
            f"p50={summary['tokens']['output_tokens']['p50']:.3f} "
            f"p95={summary['tokens']['output_tokens']['p95']:.3f}"
        ),
    ]
    (out_dir / "subset_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Done. subset summaries written under: {out_dir}")


if __name__ == "__main__":
    main()
