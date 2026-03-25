#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_by_path(payload: dict[str, Any], path: str) -> tuple[bool, Any]:
    cur: Any = payload
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def _resolve_metric(payload: dict[str, Any], candidates: list[str]) -> tuple[float | None, str | None]:
    for path in candidates:
        ok, value = _get_by_path(payload, path)
        if not ok:
            continue
        fv = _safe_float(value)
        if fv is None:
            continue
        return float(fv), path
    return None, None


def _resolve_summary_path(path: str, path_glob: str) -> Path:
    if path and path_glob:
        raise ValueError("Use either --acbv2-summary or --acbv2-summary-glob, not both")
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"ACBv2 summary not found: {p}")
        return p
    if path_glob:
        matches = sorted(glob.glob(path_glob))
        if len(matches) != 1:
            raise RuntimeError(
                f"ACBv2 summary glob must match exactly one file, got {len(matches)} for: {path_glob}"
            )
        return Path(matches[0])
    raise ValueError("Provide --acbv2-summary or --acbv2-summary-glob")


def _collect_metrics(payload: dict[str, Any]) -> tuple[dict[str, float | None], dict[str, str | None]]:
    metrics: dict[str, float | None] = {}
    paths: dict[str, str | None] = {}

    lookup = {
        "EM": ["generation.EM", "generation.em"],
        "F1": ["generation.F1", "generation.f1"],
        "ttft_p95_ms": ["latency_ms.ttft_ms.p95", "latency.ttft_ms.p95"],
        "ttft_p50_ms": ["latency_ms.ttft_ms.p50", "latency.ttft_ms.p50"],
        "t_total_p95_ms": ["latency_ms.t_total_ms.p95", "latency.t_total_ms.p95"],
        "t_total_p50_ms": ["latency_ms.t_total_ms.p50", "latency.t_total_ms.p50"],
        "context_tokens_p95": ["tokens.context_tokens.p95"],
        "context_tokens_p50": ["tokens.context_tokens.p50"],
        "pct_low_branch": ["context_budgeting.pct_low_branch"],
        "pct_medium_branch": ["context_budgeting.pct_medium_branch"],
        "pct_high_branch": ["context_budgeting.pct_high_branch"],
        "pct_fallback_to_high": ["context_budgeting.pct_fallback_to_high", "post_context.pct_fallback_to_high"],
        "support_doc_in_context_at_2_mean": ["post_context.support_doc_in_context_at_2_mean"],
        "pair_in_context_at_k_mean": ["post_context.pair_in_context_at_k_mean"],
    }

    for key, candidates in lookup.items():
        value, path = _resolve_metric(payload, candidates)
        metrics[key] = value
        paths[key] = path

    return metrics, paths


def _fmt(v: float | None, digits: int = 4) -> str:
    if v is None:
        return "null"
    return f"{v:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate report for ACB v2 vs static/acbv1 subsets")
    parser.add_argument("--static-subset", required=True)
    parser.add_argument("--acbv1-subset", required=True)
    parser.add_argument("--acbv2-summary", default="")
    parser.add_argument("--acbv2-summary-glob", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    static_path = Path(args.static_subset)
    acbv1_path = Path(args.acbv1_subset)
    acbv2_path = _resolve_summary_path(args.acbv2_summary, args.acbv2_summary_glob)

    static = _load_json(static_path)
    acbv1 = _load_json(acbv1_path)
    acbv2 = _load_json(acbv2_path)

    m_a, p_a = _collect_metrics(static)
    m_b, p_b = _collect_metrics(acbv1)
    m_c, p_c = _collect_metrics(acbv2)

    f1_a = m_a["F1"]
    f1_b = m_b["F1"]
    f1_c = m_c["F1"]
    ttft_b = m_b["ttft_p95_ms"]
    ttft_c = m_c["ttft_p95_ms"]

    gate_checks: list[tuple[str, bool, str]] = []

    cond_f1_gain = (f1_c is not None and f1_b is not None and f1_c >= (f1_b + 0.004))
    gate_checks.append(
        (
            "F1 improvement vs ACBv1",
            bool(cond_f1_gain),
            f"need F1_C >= F1_B + 0.004 | C={_fmt(f1_c)} B={_fmt(f1_b)}",
        )
    )

    cond_f1_close = (f1_c is not None and f1_a is not None and (f1_a - f1_c) <= 0.003)
    gate_checks.append(
        (
            "F1 closeness to STATIC",
            bool(cond_f1_close),
            f"need F1_A - F1_C <= 0.003 | A={_fmt(f1_a)} C={_fmt(f1_c)}",
        )
    )

    cond_ttft = False
    cond_detail = "ttft data missing"
    if ttft_c is not None and ttft_b is not None:
        cond_ttft = bool(ttft_c <= 11000.0 or ttft_c <= (1.10 * ttft_b))
        cond_detail = (
            "need TTFT_C <= 11000ms OR <= 1.10 * TTFT_B "
            f"| C={_fmt(ttft_c, 1)}ms B={_fmt(ttft_b, 1)}ms"
        )
    gate_checks.append(("TTFT p95 constraint", cond_ttft, cond_detail))

    gate_pass = all(ok for _, ok, _ in gate_checks)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# ACB v2 Gate Report",
        "",
        f"- static_subset: `{static_path}`",
        f"- acbv1_subset: `{acbv1_path}`",
        f"- acbv2_summary: `{acbv2_path}`",
        "",
        "## A/B/C Metrics",
        "",
        "| Metric | STATIC_1000 | ACBv1_1000 | ACBv2_1000 |",
        "|---|---:|---:|---:|",
        f"| EM | {_fmt(m_a['EM'])} | {_fmt(m_b['EM'])} | {_fmt(m_c['EM'])} |",
        f"| F1 | {_fmt(m_a['F1'])} | {_fmt(m_b['F1'])} | {_fmt(m_c['F1'])} |",
        f"| TTFT p50 (ms) | {_fmt(m_a['ttft_p50_ms'],1)} | {_fmt(m_b['ttft_p50_ms'],1)} | {_fmt(m_c['ttft_p50_ms'],1)} |",
        f"| TTFT p95 (ms) | {_fmt(m_a['ttft_p95_ms'],1)} | {_fmt(m_b['ttft_p95_ms'],1)} | {_fmt(m_c['ttft_p95_ms'],1)} |",
        f"| t_total p50 (ms) | {_fmt(m_a['t_total_p50_ms'],1)} | {_fmt(m_b['t_total_p50_ms'],1)} | {_fmt(m_c['t_total_p50_ms'],1)} |",
        f"| t_total p95 (ms) | {_fmt(m_a['t_total_p95_ms'],1)} | {_fmt(m_b['t_total_p95_ms'],1)} | {_fmt(m_c['t_total_p95_ms'],1)} |",
        f"| context_tokens p50 | {_fmt(m_a['context_tokens_p50'],1)} | {_fmt(m_b['context_tokens_p50'],1)} | {_fmt(m_c['context_tokens_p50'],1)} |",
        f"| context_tokens p95 | {_fmt(m_a['context_tokens_p95'],1)} | {_fmt(m_b['context_tokens_p95'],1)} | {_fmt(m_c['context_tokens_p95'],1)} |",
        f"| support_doc_in_context@2 mean | null | null | {_fmt(m_c['support_doc_in_context_at_2_mean'])} |",
        f"| pair_in_context@K mean | null | null | {_fmt(m_c['pair_in_context_at_k_mean'])} |",
        f"| pct_low_branch | null | null | {_fmt(m_c['pct_low_branch'])} |",
        f"| pct_medium_branch | null | null | {_fmt(m_c['pct_medium_branch'])} |",
        f"| pct_high_branch | null | null | {_fmt(m_c['pct_high_branch'])} |",
        f"| pct_fallback_to_high | null | null | {_fmt(m_c['pct_fallback_to_high'])} |",
        "",
        "## Gate Checks",
        "",
    ]
    for name, ok, detail in gate_checks:
        lines.append(f"- [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    lines.extend(
        [
            "",
            f"## Final Decision: {'PASS' if gate_pass else 'FAIL'}",
            "",
            "## Resolved Metric Paths",
            "",
            "### STATIC_1000",
            "",
        ]
    )
    for k, v in sorted(p_a.items()):
        lines.append(f"- {k}: `{v}`")

    lines.extend(["", "### ACBv1_1000", ""])
    for k, v in sorted(p_b.items()):
        lines.append(f"- {k}: `{v}`")

    lines.extend(["", "### ACBv2_1000", ""])
    for k, v in sorted(p_c.items()):
        lines.append(f"- {k}: `{v}`")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Done. gate report written to: {out_path}")


if __name__ == "__main__":
    main()
