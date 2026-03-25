#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RUNS = {
    "hotpot_full7405": Path(
        "results/legacy_acbsc_full_hotpot/legacy_acbsc_hotpot_full7405/"
        "cfg_32831c771548/hotpot_qa/per_query.jsonl"
    ),
    "twowiki_large6288": Path(
        "results/legacy_acbsc_large_twowiki/legacy_acbsc_twowiki_large6288/"
        "cfg_da394a03c05b/two_wiki_multihop/per_query.jsonl"
    ),
}


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    if window <= 1:
        return values.copy()
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _windowed_event_rate(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_mean(values.astype(float), window)


def _as_float(value) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _nan_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return None
    av = a[mask]
    bv = b[mask]
    if np.allclose(av, av[0]) or np.allclose(bv, bv[0]):
        return None
    return float(np.corrcoef(av, bv)[0, 1])


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=float), q))


def load_run(per_query_path: Path) -> dict:
    rows = []
    with per_query_path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Empty per_query file: {per_query_path}")

    first = rows[0]
    warmup_queries = int(first["context_budgeting"].get("warmup_queries", 0))

    positions = []
    ttft_ms = []
    total_ms = []
    prefill_ms = []
    context_tokens = []
    budget_cap_tokens = []
    cap_utilization = []
    k_eff = []
    theta_q = []
    ewma_ms_per_token = []
    dynamic_floor_tokens = []
    unlock_min_items = []
    selected_doc_count = []
    selected_count = []
    extra_probe_candidates = []
    marginal_snippets_applied = []
    pair_in_context = []
    support_doc_in_context = []
    answer_f1 = []
    answer_em = []
    power_mean_watts = []
    budget_cap_hit = []
    budget_cap_source_ewma = []
    pack_elapsed_ms = []

    fallback_reason_counts = Counter()
    budget_cap_source_counts = Counter()
    power_status_counts = Counter()

    for idx, row in enumerate(rows, start=1):
        cb = row["context_budgeting"]
        trace = cb.get("agnostic_controller_trace", {})
        post_context = row.get("post_context", {})
        answer = row.get("answer_metrics_per_query", {})
        resources = row.get("resources", {})
        lat = row["latency_ms"]

        positions.append(idx)
        ttft_ms.append(_as_float(lat.get("ttft_ms")))
        total_ms.append(_as_float(lat.get("t_total_ms")))
        prefill_ms.append(_as_float(lat.get("t_prefill_ms")))
        context_tokens.append(_as_float(cb.get("context_tokens_used")))
        budget_cap = _as_float(cb.get("budget_cap_tokens"))
        budget_cap_tokens.append(budget_cap)
        cap_utilization.append(
            float(cb.get("context_tokens_used", 0.0)) / budget_cap
            if budget_cap and math.isfinite(budget_cap) and budget_cap > 0
            else float("nan")
        )
        k_eff.append(_as_float(cb.get("k_eff")))
        theta_q.append(_as_float(cb.get("query_local_theta")))
        ewma_ms_per_token.append(_as_float(cb.get("ewma_prefill_ms_per_token")))
        dynamic_floor_tokens.append(_as_float(trace.get("dynamic_floor_tokens")))
        unlock_min_items.append(_as_float(trace.get("unlock_min_items")))
        selected_doc_count.append(_as_float(trace.get("selected_doc_count")))
        selected_count.append(_as_float(trace.get("selected_count")))
        extra_probe_candidates.append(_as_float(trace.get("extra_probe_candidates")))
        marginal_snippets_applied.append(_as_float(trace.get("marginal_snippets_applied")))
        pair_in_context.append(_as_float(post_context.get("pair_in_context_at_k")))
        support_doc_in_context.append(_as_float(post_context.get("support_doc_in_context_at_2")))
        answer_f1.append(_as_float(answer.get("f1")))
        answer_em.append(_as_float(answer.get("em")))
        power_mean_watts.append(_as_float(resources.get("power_mean_watts")))
        pack_elapsed_ms.append(_as_float(cb.get("pack_elapsed_ms")))

        fallback_reason = cb.get("fallback_reason", "")
        budget_cap_source = cb.get("budget_cap_source", "")
        power_status = resources.get("power_status", "")
        fallback_reason_counts[fallback_reason] += 1
        budget_cap_source_counts[budget_cap_source] += 1
        power_status_counts[power_status] += 1
        budget_cap_hit.append(1.0 if fallback_reason == "budget_cap" else 0.0)
        budget_cap_source_ewma.append(1.0 if budget_cap_source == "ewma_prefill" else 0.0)

    return {
        "name": per_query_path.parent.name,
        "dataset": first["dataset"],
        "split": first["split"],
        "run_id": first["run_id"],
        "config_id": first["config_id"],
        "per_query_path": str(per_query_path),
        "warmup_queries": warmup_queries,
        "positions": np.asarray(positions, dtype=float),
        "ttft_ms": np.asarray(ttft_ms, dtype=float),
        "total_ms": np.asarray(total_ms, dtype=float),
        "prefill_ms": np.asarray(prefill_ms, dtype=float),
        "context_tokens": np.asarray(context_tokens, dtype=float),
        "budget_cap_tokens": np.asarray(budget_cap_tokens, dtype=float),
        "cap_utilization": np.asarray(cap_utilization, dtype=float),
        "k_eff": np.asarray(k_eff, dtype=float),
        "theta_q": np.asarray(theta_q, dtype=float),
        "ewma_ms_per_token": np.asarray(ewma_ms_per_token, dtype=float),
        "dynamic_floor_tokens": np.asarray(dynamic_floor_tokens, dtype=float),
        "unlock_min_items": np.asarray(unlock_min_items, dtype=float),
        "selected_doc_count": np.asarray(selected_doc_count, dtype=float),
        "selected_count": np.asarray(selected_count, dtype=float),
        "extra_probe_candidates": np.asarray(extra_probe_candidates, dtype=float),
        "marginal_snippets_applied": np.asarray(marginal_snippets_applied, dtype=float),
        "pair_in_context": np.asarray(pair_in_context, dtype=float),
        "support_doc_in_context": np.asarray(support_doc_in_context, dtype=float),
        "answer_f1": np.asarray(answer_f1, dtype=float),
        "answer_em": np.asarray(answer_em, dtype=float),
        "power_mean_watts": np.asarray(power_mean_watts, dtype=float),
        "budget_cap_hit": np.asarray(budget_cap_hit, dtype=float),
        "budget_cap_source_ewma": np.asarray(budget_cap_source_ewma, dtype=float),
        "pack_elapsed_ms": np.asarray(pack_elapsed_ms, dtype=float),
        "fallback_reason_counts": dict(fallback_reason_counts),
        "budget_cap_source_counts": dict(budget_cap_source_counts),
        "power_status_counts": dict(power_status_counts),
    }


def summarize_run(run: dict) -> dict:
    n = int(len(run["positions"]))
    warmup = int(run["warmup_queries"])
    compare_n = min(500, max(100, n // 12))
    post_start = warmup
    first_slice = slice(post_start, min(n, post_start + compare_n))
    last_slice = slice(max(post_start, n - compare_n), n)

    def block_stats(arr: np.ndarray, block: slice) -> dict:
        values = arr[block]
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return {"mean": None, "median": None}
        return {"mean": float(values.mean()), "median": float(np.median(values))}

    summary = {
        "dataset": run["dataset"],
        "run_id": run["run_id"],
        "config_id": run["config_id"],
        "n_queries": n,
        "warmup_queries": warmup,
        "compare_window_queries": compare_n,
        "fallback_reason_counts": run["fallback_reason_counts"],
        "budget_cap_source_counts": run["budget_cap_source_counts"],
        "power_status_counts": run["power_status_counts"],
        "means": {
            "ttft_ms": float(np.nanmean(run["ttft_ms"])),
            "t_total_ms": float(np.nanmean(run["total_ms"])),
            "context_tokens_used": float(np.nanmean(run["context_tokens"])),
            "budget_cap_tokens": float(np.nanmean(run["budget_cap_tokens"])),
            "cap_utilization": float(np.nanmean(run["cap_utilization"])),
            "k_eff": float(np.nanmean(run["k_eff"])),
            "theta_q": float(np.nanmean(run["theta_q"])),
            "ewma_prefill_ms_per_token": float(np.nanmean(run["ewma_ms_per_token"])),
            "dynamic_floor_tokens": float(np.nanmean(run["dynamic_floor_tokens"])),
            "unlock_min_items": float(np.nanmean(run["unlock_min_items"])),
            "selected_doc_count": float(np.nanmean(run["selected_doc_count"])),
            "extra_probe_candidates": float(np.nanmean(run["extra_probe_candidates"])),
            "marginal_snippets_applied": float(np.nanmean(run["marginal_snippets_applied"])),
            "pair_in_context": float(np.nanmean(run["pair_in_context"])),
            "support_doc_in_context_at_2": float(np.nanmean(run["support_doc_in_context"])),
            "power_mean_watts": float(np.nanmean(run["power_mean_watts"])),
        },
        "medians": {
            "ttft_ms": float(np.nanmedian(run["ttft_ms"])),
            "context_tokens_used": float(np.nanmedian(run["context_tokens"])),
            "budget_cap_tokens": float(np.nanmedian(run["budget_cap_tokens"])),
            "cap_utilization": float(np.nanmedian(run["cap_utilization"])),
            "k_eff": float(np.nanmedian(run["k_eff"])),
        },
        "event_rates": {
            "budget_cap_hit_rate": float(np.nanmean(run["budget_cap_hit"])),
            "marginal_snippet_rate": float(np.nanmean(run["marginal_snippets_applied"] > 0)),
            "ewma_budget_rate": float(np.nanmean(run["budget_cap_source_ewma"])),
        },
        "correlations": {
            "context_tokens_vs_ttft": _nan_corr(run["context_tokens"], run["ttft_ms"]),
            "budget_cap_vs_ttft": _nan_corr(run["budget_cap_tokens"], run["ttft_ms"]),
            "cap_utilization_vs_pair_in_context": _nan_corr(run["cap_utilization"], run["pair_in_context"]),
            "theta_q_vs_context_tokens": _nan_corr(run["theta_q"], run["context_tokens"]),
            "ewma_ms_per_token_vs_ttft": _nan_corr(run["ewma_ms_per_token"], run["ttft_ms"]),
        },
        "early_late": {
            "first_post_warmup": {
                "query_range": [post_start + 1, min(n, post_start + compare_n)],
                "ttft_ms": block_stats(run["ttft_ms"], first_slice),
                "context_tokens_used": block_stats(run["context_tokens"], first_slice),
                "budget_cap_tokens": block_stats(run["budget_cap_tokens"], first_slice),
                "cap_utilization": block_stats(run["cap_utilization"], first_slice),
                "k_eff": block_stats(run["k_eff"], first_slice),
                "theta_q": block_stats(run["theta_q"], first_slice),
                "ewma_prefill_ms_per_token": block_stats(run["ewma_ms_per_token"], first_slice),
                "pair_in_context": block_stats(run["pair_in_context"], first_slice),
                "support_doc_in_context_at_2": block_stats(run["support_doc_in_context"], first_slice),
            },
            "last_window": {
                "query_range": [max(post_start, n - compare_n) + 1, n],
                "ttft_ms": block_stats(run["ttft_ms"], last_slice),
                "context_tokens_used": block_stats(run["context_tokens"], last_slice),
                "budget_cap_tokens": block_stats(run["budget_cap_tokens"], last_slice),
                "cap_utilization": block_stats(run["cap_utilization"], last_slice),
                "k_eff": block_stats(run["k_eff"], last_slice),
                "theta_q": block_stats(run["theta_q"], last_slice),
                "ewma_prefill_ms_per_token": block_stats(run["ewma_ms_per_token"], last_slice),
                "pair_in_context": block_stats(run["pair_in_context"], last_slice),
                "support_doc_in_context_at_2": block_stats(run["support_doc_in_context"], last_slice),
            },
        },
    }
    return summary


def plot_run(run: dict, summary: dict, out_path: Path) -> None:
    _ensure_parent(out_path)
    n = int(len(run["positions"]))
    warmup = int(run["warmup_queries"])
    window = max(51, min(151, ((n // 75) | 1)))
    x = run["positions"]

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
    axes = axes.ravel()
    title = (
        f"{run['dataset']} | ACB-SC temporal behavior | "
        f"n={n}, warmup={warmup}, rolling window={window}"
    )
    fig.suptitle(title, fontsize=14)

    def shade_warmup(ax):
        if warmup > 0:
            ax.axvspan(1, warmup, color="#f0f0f0", alpha=0.75, lw=0)
            ax.axvline(warmup, color="#888888", ls="--", lw=1)

    # 1. Latency
    ax = axes[0]
    ax.plot(x, _rolling_mean(run["ttft_ms"], window), label="TTFT (rolling mean)", color="#1f77b4")
    ax.plot(x, _rolling_mean(run["prefill_ms"], window), label="Prefill (rolling mean)", color="#17becf", alpha=0.8)
    ax.plot(x, _rolling_mean(run["total_ms"], window), label="Total latency (rolling mean)", color="#ff7f0e")
    target = float(run["budget_cap_tokens"][0] * run["ewma_ms_per_token"][warmup]) if n > warmup and np.isfinite(run["ewma_ms_per_token"][warmup]) else 18000.0
    ax.axhline(18000.0, color="#d62728", ls=":", lw=1.5, label="Prefill target")
    shade_warmup(ax)
    ax.set_title("Latency envelope over query index")
    ax.set_xlabel("Query index")
    ax.set_ylabel("ms")
    ax.legend(fontsize=8, loc="upper right")

    # 2. Token budget vs actual usage
    ax = axes[1]
    ax.plot(x, _rolling_mean(run["budget_cap_tokens"], window), label="Budget cap tokens", color="#2ca02c")
    ax.plot(x, _rolling_mean(run["context_tokens"], window), label="Context tokens used", color="#9467bd")
    ax.plot(x, _rolling_mean(run["dynamic_floor_tokens"], window), label="Dynamic floor tokens", color="#8c564b")
    shade_warmup(ax)
    ax.set_title("Adaptive token budget vs realized context")
    ax.set_xlabel("Query index")
    ax.set_ylabel("tokens")
    ax.legend(fontsize=8, loc="upper right")

    # 3. Utilization + k_eff
    ax = axes[2]
    ax.plot(x, _rolling_mean(run["cap_utilization"], window), label="Cap utilization", color="#1f77b4")
    ax.set_ylabel("context / cap")
    ax.set_xlabel("Query index")
    ax.set_title("How much of the adaptive cap is actually used")
    ax.set_ylim(bottom=0.0)
    shade_warmup(ax)
    ax2 = ax.twinx()
    ax2.plot(x, _rolling_mean(run["k_eff"], window), label="k_eff", color="#ff7f0e")
    ax2.plot(x, _rolling_mean(run["selected_doc_count"], window), label="selected docs", color="#2ca02c", alpha=0.85)
    ax2.set_ylabel("items / docs")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper right")

    # 4. Controller thresholds and hardware-adaptive rate
    ax = axes[3]
    ax.plot(x, _rolling_mean(run["theta_q"], window), label="theta_q", color="#9467bd")
    ax.set_ylabel("query-local threshold")
    ax.set_xlabel("Query index")
    ax.set_title("Query-local threshold and EWMA prefill rate")
    shade_warmup(ax)
    ax2 = ax.twinx()
    ax2.plot(x, _rolling_mean(run["ewma_ms_per_token"], window), label="EWMA ms/token", color="#d62728")
    ax2.set_ylabel("ms/token")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper right")

    # 5. Final context quality proxies
    ax = axes[4]
    ax.plot(x, _rolling_mean(run["pair_in_context"], window), label="pair_in_context@k", color="#2ca02c")
    ax.plot(x, _rolling_mean(run["support_doc_in_context"], window), label="support_doc_in_context@2", color="#17becf")
    shade_warmup(ax)
    ax.set_title("Final-context quality over time")
    ax.set_xlabel("Query index")
    ax.set_ylabel("rate")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8, loc="lower right")

    # 6. Controller actions
    ax = axes[5]
    ax.plot(x, _windowed_event_rate(run["marginal_snippets_applied"] > 0, window), label="marginal snippet rate", color="#8c564b")
    ax.plot(x, _windowed_event_rate(run["budget_cap_hit"] > 0, window), label="budget-cap stop rate", color="#d62728")
    ax.plot(x, _rolling_mean(run["extra_probe_candidates"], window), label="extra probe candidates", color="#7f7f7f")
    ax.plot(x, _windowed_event_rate(run["budget_cap_source_ewma"] > 0, window), label="EWMA-cap source rate", color="#bcbd22")
    shade_warmup(ax)
    ax.set_title("Control actions and warmup transition")
    ax.set_xlabel("Query index")
    ax.set_ylabel("rate / count")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=8, loc="upper right")

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_markdown_report(summaries: dict[str, dict], out_path: Path) -> None:
    _ensure_parent(out_path)
    lines = [
        "# ACB-SC Temporal Analysis",
        "",
        "This report summarizes controller-specific temporal behavior for the long ACB-SC runs.",
        "",
    ]

    for key, summary in summaries.items():
        lines.append(f"## {key}")
        lines.append("")
        lines.append(f"- Dataset: `{summary['dataset']}`")
        lines.append(f"- Queries: `{summary['n_queries']}`")
        lines.append(f"- Warmup queries: `{summary['warmup_queries']}`")
        lines.append(f"- Comparison window: first/last `{summary['compare_window_queries']}` post-warmup queries")
        lines.append("")
        lines.append("### Controller summary")
        lines.append("")
        lines.append(
            f"- Mean context vs cap: `{summary['means']['context_tokens_used']:.1f}` / "
            f"`{summary['means']['budget_cap_tokens']:.1f}` tokens "
            f"(`{100.0 * summary['means']['cap_utilization']:.1f}%` utilization)"
        )
        lines.append(
            f"- Mean `k_eff`: `{summary['means']['k_eff']:.2f}`; mean selected docs: "
            f"`{summary['means']['selected_doc_count']:.2f}`"
        )
        lines.append(
            f"- Mean `theta_q`: `{summary['means']['theta_q']:.3f}`; mean EWMA rate: "
            f"`{summary['means']['ewma_prefill_ms_per_token']:.2f}` ms/token"
        )
        lines.append(
            f"- Mean dynamic floor: `{summary['means']['dynamic_floor_tokens']:.1f}` tokens; "
            f"mean extra probes: `{summary['means']['extra_probe_candidates']:.2f}`"
        )
        lines.append(
            f"- Mean TTFT: `{summary['means']['ttft_ms']:.1f}` ms; median TTFT: "
            f"`{summary['medians']['ttft_ms']:.1f}` ms"
        )
        lines.append(
            f"- Mean final-context quality: `pair_in_context={summary['means']['pair_in_context']:.3f}`, "
            f"`support_doc_in_context_at_2={summary['means']['support_doc_in_context_at_2']:.3f}`"
        )
        lines.append(
            f"- Event rates: `budget_cap_hit={100.0 * summary['event_rates']['budget_cap_hit_rate']:.3f}%`, "
            f"`marginal_snippet={100.0 * summary['event_rates']['marginal_snippet_rate']:.1f}%`, "
            f"`ewma_cap_source={100.0 * summary['event_rates']['ewma_budget_rate']:.1f}%`"
        )
        lines.append(
            f"- Fallback reasons: `{json.dumps(summary['fallback_reason_counts'], ensure_ascii=True)}`"
        )
        lines.append(
            f"- Budget cap sources: `{json.dumps(summary['budget_cap_source_counts'], ensure_ascii=True)}`"
        )
        lines.append("")
        lines.append("### Early vs late behavior")
        lines.append("")
        early = summary["early_late"]["first_post_warmup"]
        late = summary["early_late"]["last_window"]
        for metric in [
            "ttft_ms",
            "context_tokens_used",
            "budget_cap_tokens",
            "cap_utilization",
            "k_eff",
            "theta_q",
            "ewma_prefill_ms_per_token",
            "pair_in_context",
            "support_doc_in_context_at_2",
        ]:
            e = early[metric]["mean"]
            l = late[metric]["mean"]
            if e is None or l is None:
                continue
            delta = l - e
            pct = 100.0 * delta / e if e else float("nan")
            lines.append(
                f"- `{metric}`: `{e:.3f}` -> `{l:.3f}` "
                f"(Δ `{delta:+.3f}`, `{pct:+.1f}%`)"
            )
        lines.append("")
        lines.append("### Correlations")
        lines.append("")
        for metric, value in summary["correlations"].items():
            if value is None:
                continue
            lines.append(f"- `{metric}`: `{value:+.3f}`")
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n")


def write_json(summaries: dict[str, dict], out_path: Path) -> None:
    _ensure_parent(out_path)
    out_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="results/final_full_runs/acbsc_temporal_analysis",
        help="Directory for figures and report",
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("NAME", "PER_QUERY_JSONL"),
        help="Optional extra run to analyze",
    )
    args = parser.parse_args()

    runs = dict(DEFAULT_RUNS)
    if args.run:
        for name, path in args.run:
            runs[name] = Path(path)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for name, per_query_path in runs.items():
        run = load_run(per_query_path)
        summary = summarize_run(run)
        summaries[name] = summary
        plot_run(run, summary, out_dir / f"{name}_overview.png")

    build_markdown_report(summaries, out_dir / "README.md")
    write_json(summaries, out_dir / "summary.json")

    print(json.dumps({
        "output_dir": str(out_dir),
        "runs": list(runs.keys()),
        "artifacts": [
            str(out_dir / "README.md"),
            str(out_dir / "summary.json"),
            *[str(out_dir / f"{name}_overview.png") for name in runs],
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
