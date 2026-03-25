from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from .chunking import chunk_documents
from .context_budgeting import resolve_context_budgeting_config
from .data import (
    build_qrels_from_qa_examples,
    load_beir_dataset,
    load_hotpotqa_distractor,
    load_natural_questions_validation,
    load_two_wiki_multihop_validation,
    map_doc_qrels_to_chunk_qrels,
)
from .metrics import evaluate_retrieval, qa_scores, summarize_list
from .pipeline import RAGPipeline
from .profiling import QueryResourceSampler
from .retrievers import chunks_to_items, docs_to_items, unique_doc_ids
from .runtime_profiles import resolve_llm_runtime
from .utils import save_json


DEFAULT_RETRIEVAL_KS = [1, 2, 5, 10, 20, 50, 100]


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(round((len(s) - 1) * q))
    idx = min(len(s) - 1, max(0, idx))
    return float(s[idx])


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _select_stats_values(values: list[float], positive_only: bool = False) -> list[float]:
    if not positive_only:
        return list(values)
    filtered = [float(v) for v in values if float(v) > 0.0]
    return filtered if filtered else list(values)


def _format_mean_p50_p95(
    values: list[float],
    unit: str = "ms",
    positive_only: bool = False,
    scale: float = 1.0,
) -> str:
    src = _select_stats_values(values, positive_only=positive_only)
    if not src:
        return f"0.0/0.0/0.0{unit}"
    m = _mean(src) / max(scale, 1e-12)
    p50 = _pct(src, 0.5) / max(scale, 1e-12)
    p95 = _pct(src, 0.95) / max(scale, 1e-12)
    return f"{m:.1f}/{p50:.1f}/{p95:.1f}{unit}"


def _format_rate(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.0%"
    return f"{(100.0 * float(numerator) / float(denominator)):.1f}%"


def _format_rss_cpu(rss_peak_bytes_values: list[float], cpu_mean_pct_values: list[float]) -> str:
    rss_mb = _mean(rss_peak_bytes_values) / (1024.0 * 1024.0)
    cpu_mean = _mean(cpu_mean_pct_values)
    return f"{rss_mb:.0f}MB/{cpu_mean:.1f}%"


def _format_power(power_mean_watts_values: list[float], power_peak_watts_values: list[float]) -> str:
    mean_vals = _select_stats_values(power_mean_watts_values, positive_only=True)
    peak_vals = _select_stats_values(power_peak_watts_values, positive_only=True)
    if not mean_vals and not peak_vals:
        return "n/a"
    mean_w = _mean(mean_vals) if mean_vals else 0.0
    peak_w = _pct(peak_vals, 0.95) if peak_vals else 0.0
    return f"{mean_w:.2f}/{peak_w:.2f}W"


def _qa_status_table(
    em_values: list[float],
    f1_values: list[float],
    valid_answer_count: int,
    failure_count: int,
    processed: int,
    t_total_vals: list[float],
    ttft_vals: list[float],
    tps_decode_vals: list[float],
    rss_peak_vals: list[float],
    cpu_mean_vals: list[float],
    power_mean_vals: list[float],
    power_peak_vals: list[float],
) -> Table:
    table = Table.grid(expand=True)
    table.add_column(ratio=1, justify="left")
    table.add_column(ratio=1, justify="left")
    table.add_row(
        f"EM/F1: {_mean(em_values):.3f}/{_mean(f1_values):.3f}",
        f"valid/fail: {_format_rate(valid_answer_count, processed)}/{_format_rate(failure_count, processed)}",
    )
    table.add_row(
        f"t_total mean/p50/p95: {_format_mean_p50_p95(t_total_vals, unit='ms', positive_only=True)}",
        f"ttft mean/p50/p95: {_format_mean_p50_p95(ttft_vals, unit='ms', positive_only=True)}",
    )
    tps_mean = _mean(_select_stats_values(tps_decode_vals, positive_only=True))
    table.add_row(
        f"decode tok/s (mean): {tps_mean:.1f}",
        f"RSS peak / CPU mean: {_format_rss_cpu(rss_peak_vals, cpu_mean_vals)}",
    )
    table.add_row(
        f"power mean/p95-peak: {_format_power(power_mean_vals, power_peak_vals)}",
        "",
    )
    return table


def _beir_status_table(
    recall10_vals: list[float],
    ndcg10_vals: list[float],
    failure_count: int,
    processed: int,
    t_total_vals: list[float],
    rss_peak_vals: list[float],
    cpu_mean_vals: list[float],
    power_mean_vals: list[float],
    power_peak_vals: list[float],
) -> Table:
    table = Table.grid(expand=True)
    table.add_column(ratio=1, justify="left")
    table.add_column(ratio=1, justify="left")
    table.add_row(
        f"R@10/nDCG@10: {_mean(recall10_vals):.3f}/{_mean(ndcg10_vals):.3f}",
        f"fail: {_format_rate(failure_count, processed)}",
    )
    table.add_row(
        f"t_total mean/p50/p95: {_format_mean_p50_p95(t_total_vals, unit='ms', positive_only=True)}",
        f"RSS peak / CPU mean: {_format_rss_cpu(rss_peak_vals, cpu_mean_vals)}",
    )
    table.add_row(
        f"power mean/p95-peak: {_format_power(power_mean_vals, power_peak_vals)}",
        "",
    )
    return table


def _empty_token_stats() -> dict[str, float]:
    return {
        "context_tokens": 0.0,
        "output_tokens": 0.0,
        "total_tokens": 0.0,
        "tokens_per_second_decode": 0.0,
        "tokens_per_second_prefill": 0.0,
    }


def _empty_context_budgeting(cfg: dict[str, Any]) -> dict[str, Any]:
    cb = cfg.get("context_budgeting", {})
    strategy = str(cb.get("strategy", "v1")).strip().lower()
    resolved_v2 = {}
    if strategy == "v2_evidence_first":
        resolved_v2 = {
            "keep_full_count": int(cb.get("keep_full_count", 2)),
            "mmr_lambda": float(cb.get("mmr_lambda", 0.30)),
            "min_low_savings_ratio": float(cb.get("min_low_savings_ratio", 0.15)),
            "k_eff_floor": int(cb.get("k_eff_floor", cb.get("k_low", 5))),
            "snippet_from_rank": int(cb.get("snippet_from_rank", 3)),
            "low_margin_multiplier": float(cb.get("low_margin_multiplier", 5.0)),
            "low_agreement_multiplier": float(cb.get("low_agreement_multiplier", 1.0)),
            "medium_branch_enabled": bool(cb.get("medium_branch_enabled", True)),
            "medium_budget_tokens": int(cb.get("medium_budget_tokens", 900)),
            "medium_k_eff": int(cb.get("medium_k_eff", 8)),
            "top_doc_saliency_tokens": int(cb.get("top_doc_saliency_tokens", 192)),
            "saliency_entity_weight": float(cb.get("saliency_entity_weight", 2.0)),
            "dynamic_mmr_enabled": bool(cb.get("dynamic_mmr_enabled", True)),
            "dynamic_mmr_threshold": float(cb.get("dynamic_mmr_threshold", 0.45)),
            "dynamic_mmr_boost": float(cb.get("dynamic_mmr_boost", 0.60)),
            "dynamic_mmr_cap": float(cb.get("dynamic_mmr_cap", 1.20)),
        }
    return {
        "enabled": bool(cb.get("enabled", False)),
        "strategy": strategy,
        "k_eff": 0,
        "context_budget_tokens": 0,
        "context_tokens_used": 0,
        "policy_branch": "disabled",
        "margin_value": None,
        "agreement_value": None,
        "margin_source": "none",
        "margin_threshold": float(cb.get("margin_threshold_fallback", 0.003)),
        "agreement_threshold": float(cb.get("agreement_threshold", 0.35)),
        "margin_threshold_source": "fallback",
        "margin_threshold_quantile": float(cb.get("margin_threshold_quantile", 0.9)),
        "margin_threshold_stage2_glob": str(cb.get("margin_threshold_stage2_glob", "")),
        "margin_threshold_retriever_mode": str(cb.get("margin_threshold_stage2_retriever_mode", "hybrid")),
        "margin_threshold_samples": 0,
        "keep_full_count": int(cb.get("keep_full_count", 0)),
        "fallback_to_high": False,
        "fallback_reason": "",
        "context_chunk_ids_used": [],
        "context_doc_ids_used": [],
        "context_doc_internal_ids_used": [],
        "redundancy_avg_similarity": 0.0,
        "redundancy_max_similarity": 0.0,
        "mmr_lambda_max_used": float(cb.get("mmr_lambda", 0.30)),
        "aliases_used": {},
        "unknown_keys": [],
        "v2_resolved_params": resolved_v2,
    }


def _empty_sp3(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(resolve_llm_runtime(cfg))


def _empty_post_context() -> dict[str, Any]:
    return {
        "context_chunk_ids_used": [],
        "context_doc_ids_used": [],
        "context_doc_internal_ids_used": [],
        "keep_full_count": 0,
        "fallback_to_high": False,
        "fallback_reason": "",
        "redundancy_avg_similarity": 0.0,
        "redundancy_max_similarity": 0.0,
        "mmr_lambda_max_used": 0.0,
        "support_doc_in_context_at_2": None,
        "pair_in_context_at_k": None,
    }


def _empty_latency(include_llm: bool = True) -> dict[str, float]:
    latency = {
        "t_retrieval_total_ms": 0.0,
        "t_bm25_search_ms": 0.0,
        "t_query_embed_ms": 0.0,
        "t_vector_search_ms": 0.0,
        "t_merge_hybrid_ms": 0.0,
        "t_rerank_total_ms": 0.0,
        "t_total_ms": 0.0,
    }
    if include_llm:
        latency.update(
            {
                "t_context_pack_ms": 0.0,
                "t_prompt_build_ms": 0.0,
                "t_llm_total_ms": 0.0,
                "t_prefill_ms": 0.0,
                "ttft_ms": 0.0,
                "t_decode_total_ms": 0.0,
                "t_postprocess_ms": 0.0,
            }
        )
    return latency


def _empty_retrieval_stages(cfg: dict[str, Any]) -> dict[str, Any]:
    ret = cfg["retrieval"]
    mode = str(ret.get("retriever_mode", "hybrid")).lower()
    fusion_method = str(ret.get("fusion_method", "RRF")).upper()
    weighted_alpha = float(ret.get("weighted_alpha", ret.get("hybrid_alpha", 0.5)))
    return {
        "retriever_mode": mode,
        "fusion_method": fusion_method if mode == "hybrid" else None,
        "rrf_k": int(ret.get("rrf_k", 60)) if mode == "hybrid" and fusion_method != "WEIGHTED_SUM" else None,
        "weighted_alpha": weighted_alpha if mode == "hybrid" and fusion_method == "WEIGHTED_SUM" else None,
        "bm25": {
            "topN_ids": [],
            "topN_scores": [],
            "t_bm25_search_ms": 0.0,
        },
        "dense": {
            "topN_ids": [],
            "topN_scores": [],
            "t_query_embed_ms": 0.0,
            "t_vector_search_ms": 0.0,
        },
        "fusion": {
            "fused_topK_ids": [],
            "fused_topK_scores": [],
            "t_merge_hybrid_ms": 0.0,
        },
        "retrieval_final_topk_ids": [],
        "retrieval_final_topk_scores": [],
        "overlap_count": 0,
        "overlap_ratio": 0.0,
        "t_retrieval_total_ms": 0.0,
    }


def config_fingerprint(cfg: dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def resolve_query_budget(dataset: str, tier: str, override: int | None = None) -> int:
    if override is not None and override > 0:
        return int(override)
    t = tier.upper()
    if t == "A":
        return 200
    if t == "B":
        return 1000
    if dataset == "hotpot_qa":
        return 7405
    return 1000


def _load_query_ids(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"query ids path not found: {p}")
    raw = p.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if p.suffix.lower() == ".json":
        payload = json.loads(raw)
        if not isinstance(payload, list):
            raise ValueError(f"query ids JSON must be a list: {p}")
        return [str(x) for x in payload if str(x).strip()]
    return [line.strip() for line in raw.splitlines() if line.strip()]


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_timeseries(
    out_dir: Path,
    qid: str,
    rss_ts: list[dict[str, float]],
    cpu_ts: list[dict[str, float]],
) -> tuple[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_qid = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in qid)
    path = out_dir / f"{safe_qid}.jsonl.gz"

    rows = []
    n = max(len(rss_ts), len(cpu_ts))
    for i in range(n):
        row: dict[str, float] = {}
        if i < len(rss_ts):
            row["timestamp_ms"] = float(rss_ts[i]["timestamp_ms"])
            row["rss_bytes"] = float(rss_ts[i]["rss_bytes"])
        if i < len(cpu_ts):
            row["timestamp_ms"] = float(cpu_ts[i]["timestamp_ms"])
            row["cpu_pct"] = float(cpu_ts[i]["cpu_pct"])
        rows.append(row)

    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(path), len(rows)


def _support_doc_recall_at_ks(
    retrieved_doc_ids: list[str],
    gold_doc_ids: set[str],
    ks: list[int],
) -> dict[str, float]:
    out: dict[str, float] = {}
    if not gold_doc_ids:
        for k in ks:
            out[f"support_doc_recall@{k}"] = 0.0
        return out
    for k in ks:
        top = retrieved_doc_ids[:k]
        hits = len([d for d in top if d in gold_doc_ids])
        out[f"support_doc_recall@{k}"] = float(hits / max(1, len(gold_doc_ids)))
    return out


def _pair_recall_at_ks(
    retrieved_doc_ids: list[str],
    gold_doc_ids: set[str],
    ks: list[int],
) -> tuple[dict[str, float], bool]:
    # HotpotQA has a gold supporting pair (2 docs). Keep pair_recall defined on that regime only.
    if len(gold_doc_ids) != 2:
        return {}, False
    out: dict[str, float] = {}
    for k in ks:
        top = retrieved_doc_ids[:k]
        hits = len([d for d in top if d in gold_doc_ids])
        out[f"pair_recall@{k}"] = 1.0 if hits == 2 else 0.0
    return out, True


def _support_doc_in_context_at_2(context_doc_ids: list[str], gold_doc_ids: set[str]) -> float | None:
    if not gold_doc_ids:
        return None
    top2 = list(context_doc_ids[:2])
    hits = len([doc_id for doc_id in top2 if doc_id in gold_doc_ids])
    return float(hits / max(1, len(gold_doc_ids)))


def _pair_in_context_at_k(context_doc_ids: list[str], gold_doc_ids: set[str]) -> float | None:
    if len(gold_doc_ids) != 2:
        return None
    context_set = set(context_doc_ids)
    return 1.0 if len(context_set.intersection(gold_doc_ids)) == 2 else 0.0


def _blank_generation() -> dict[str, Any]:
    return {
        "EM": 0.0,
        "F1": 0.0,
        "rouge_l": 0.0,
        "abstention_rate": 0.0,
        "accuracy_when_answered": 0.0,
    }


def _aggregate_summary_stats(values: list[float]) -> dict[str, float]:
    return summarize_list(values)


def _make_run_manifest(
    run_id: str,
    cfg: dict[str, Any],
    dataset: str,
    split: str,
    tier: str,
    seed: int,
    num_queries: int,
    profile_timeseries: bool,
    sampling_interval_ms: int,
    timeseries_stride: int,
    profile_power: bool,
    power_sampling_interval_ms: int,
    ui_update_every: int,
    query_ids_path: str,
) -> dict[str, Any]:
    ret = cfg["retrieval"]
    llm = cfg["llm"]
    cb = cfg.get("context_budgeting", {})
    cb_resolved = resolve_context_budgeting_config(cb)
    rt = cfg.get("llm_runtime", {})
    runtime_effective = resolve_llm_runtime(cfg)
    fusion_method = str(ret.get("fusion_method", "RRF")).upper()
    return {
        "run_id": run_id,
        "config_id": config_fingerprint(cfg),
        "dataset": dataset,
        "split": split,
        "tier": tier,
        "seed": int(seed),
        "num_queries": int(num_queries),
        "query_ids_path": query_ids_path,
        "model_ids": {
            "llm_gguf": llm.get("gguf_path", ""),
            "dense_model": ret.get("dense_model", ""),
            "reranker_model": cfg["reranker"].get("model_name", ""),
        },
        "retriever_mode": ret.get("retriever_mode", "hybrid"),
        "fusion_method": fusion_method,
        "rrf_k": int(ret.get("rrf_k", 60)),
        "agreement_bonus_enabled": bool(ret.get("agreement_bonus_enabled", False)),
        "agreement_bonus": float(ret.get("agreement_bonus", 0.0)),
        "weighted_alpha": (
            float(ret.get("weighted_alpha", ret.get("hybrid_alpha", 0.5)))
            if fusion_method == "WEIGHTED_SUM"
            else None
        ),
        "context_budgeting": {
            "enabled": bool(cb_resolved.get("enabled", cb.get("enabled", False))),
            "strategy": str(cb_resolved.get("strategy", cb.get("strategy", "v1"))),
            "k_low": int(cb.get("k_low", 5)),
            "budget_low_tokens": int(cb.get("budget_low_tokens", 600)),
            "budget_high_tokens": int(cb.get("budget_high_tokens", 1200)),
            "margin_threshold": cb.get("margin_threshold", None),
            "margin_threshold_quantile": float(cb.get("margin_threshold_quantile", 0.9)),
            "margin_threshold_fallback": float(cb.get("margin_threshold_fallback", 0.003)),
            "margin_threshold_stage2_glob": str(cb.get("margin_threshold_stage2_glob", "results/*/stage2/*/per_query.jsonl")),
            "margin_threshold_stage2_retriever_mode": str(cb.get("margin_threshold_stage2_retriever_mode", "hybrid")),
            "agreement_threshold": float(cb.get("agreement_threshold", 0.35)),
            "use_rerank_scores_if_available": bool(cb.get("use_rerank_scores_if_available", True)),
            "packing_mode_low": str(cb.get("packing_mode_low", "full_or_light")),
            "packing_mode_high": str(cb.get("packing_mode_high", "snippet")),
            "snippet_window_tokens": int(cb.get("snippet_window_tokens", 80)),
            "max_chunks_hard_cap": int(cb.get("max_chunks_hard_cap", 20)),
            "v2_resolved_params": (
                {
                    "keep_full_count": int(cb_resolved.get("keep_full_count", 2)),
                    "mmr_lambda": float(cb_resolved.get("mmr_lambda", 0.30)),
                    "min_low_savings_ratio": float(cb_resolved.get("min_low_savings_ratio", 0.15)),
                    "k_eff_floor": int(cb_resolved.get("k_eff_floor", 5)),
                    "snippet_from_rank": int(cb_resolved.get("snippet_from_rank", 3)),
                    "low_margin_multiplier": float(cb_resolved.get("low_margin_multiplier", 5.0)),
                    "low_agreement_multiplier": float(cb_resolved.get("low_agreement_multiplier", 1.0)),
                    "medium_branch_enabled": bool(cb_resolved.get("medium_branch_enabled", True)),
                    "medium_budget_tokens": int(cb_resolved.get("medium_budget_tokens", 900)),
                    "medium_k_eff": int(cb_resolved.get("medium_k_eff", 8)),
                    "top_doc_saliency_tokens": int(cb_resolved.get("top_doc_saliency_tokens", 192)),
                    "saliency_entity_weight": float(cb_resolved.get("saliency_entity_weight", 2.0)),
                    "dynamic_mmr_enabled": bool(cb_resolved.get("dynamic_mmr_enabled", True)),
                    "dynamic_mmr_threshold": float(cb_resolved.get("dynamic_mmr_threshold", 0.45)),
                    "dynamic_mmr_boost": float(cb_resolved.get("dynamic_mmr_boost", 0.60)),
                    "dynamic_mmr_cap": float(cb_resolved.get("dynamic_mmr_cap", 1.20)),
                }
                if str(cb_resolved.get("strategy", "v1")) == "v2_evidence_first"
                else {}
            ),
            "aliases_used": dict(cb_resolved.get("aliases_used", {})),
            "unknown_keys": list(cb_resolved.get("unknown_keys", [])),
        },
        "sp3": {
            "enabled": bool(runtime_effective.get("enabled", False)),
            "profile": str(runtime_effective.get("profile", "BASELINE")).upper(),
            "threads_decode": int(runtime_effective.get("threads_decode", llm.get("n_threads", 4))),
            "threads_batch": int(runtime_effective.get("threads_batch", llm.get("n_threads", 4))),
            "batch_size": runtime_effective.get("batch_size", None),
            "ubatch_size": runtime_effective.get("ubatch_size", None),
        },
        "sp3_requested": {
            "enabled": bool(rt.get("sp3_enabled", False)),
            "profile": str(rt.get("sp3_profile", "BASELINE")).upper(),
            "threads_decode": rt.get("threads_decode", None),
            "threads_batch": rt.get("threads_batch", None),
            "batch_size": rt.get("batch_size", None),
            "ubatch_size": rt.get("ubatch_size", None),
        },
        "profile_timeseries": bool(profile_timeseries),
        "sampling_interval_ms": int(sampling_interval_ms),
        "timeseries_stride": int(timeseries_stride),
        "profile_power": bool(profile_power),
        "power_sampling_interval_ms": int(power_sampling_interval_ms),
        "power_backend": ("powermetrics" if profile_power else None),
        "ui_update_every": int(ui_update_every),
        "config": cfg,
    }


def run_qa_benchmark(
    cfg: dict[str, Any],
    dataset: str,
    tier: str,
    output_dir: str | Path,
    run_id: str,
    seed: int,
    max_queries_override: int | None = None,
    retrieval_ks: list[int] | None = None,
    profile_timeseries: bool = False,
    sampling_interval_ms: int = 200,
    timeseries_stride: int = 5,
    profile_power: bool | None = None,
    power_sampling_interval_ms: int | None = None,
    ui_update_every: int = 5,
    query_ids_path: str | Path | None = None,
) -> dict[str, Any]:
    if dataset not in {"hotpot_qa", "natural_questions", "two_wiki_multihop"}:
        raise ValueError(f"Unsupported QA dataset: {dataset}")

    ks = sorted(set([int(k) for k in (retrieval_ks or DEFAULT_RETRIEVAL_KS)]))
    num_queries = resolve_query_budget(dataset, tier, max_queries_override)
    split = "validation"
    profiling_cfg = cfg.get("profiling", {})
    effective_profile_power = (
        bool(profile_power) if profile_power is not None else bool(profiling_cfg.get("profile_power", False))
    )
    effective_power_sampling_interval_ms = (
        int(power_sampling_interval_ms)
        if power_sampling_interval_ms is not None
        else int(profiling_cfg.get("power_sampling_interval_ms", 1000))
    )
    forced_qids: list[str] = _load_query_ids(query_ids_path) if query_ids_path else []
    if forced_qids and num_queries > 0:
        forced_qids = forced_qids[: int(num_queries)]

    run_path = Path(output_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    per_query_path = run_path / "per_query.jsonl"
    if per_query_path.exists():
        per_query_path.unlink()
    timeseries_dir = run_path / "timeseries"

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
    ) as setup_progress:
        setup_task_id = setup_progress.add_task("QA bootstrap: loading dataset", total=None)
        if dataset == "hotpot_qa":
            examples, docs = load_hotpotqa_distractor(
                split=split,
                max_queries=num_queries,
                seed=seed,
                include_qids=(forced_qids if forced_qids else None),
            )
        elif dataset == "two_wiki_multihop":
            examples, docs = load_two_wiki_multihop_validation(
                max_queries=num_queries,
                seed=seed,
                include_qids=(forced_qids if forced_qids else None),
            )
        else:
            examples, docs = load_natural_questions_validation(
                max_queries=num_queries,
                seed=seed,
                include_qids=(forced_qids if forced_qids else None),
            )

        qid_path = run_path / "sampled_qids.json"
        save_json(qid_path, [ex.qid for ex in examples])
        manifest = _make_run_manifest(
            run_id=run_id,
            cfg=cfg,
            dataset=dataset,
            split=split,
            tier=tier,
            seed=seed,
            num_queries=len(examples),
            profile_timeseries=profile_timeseries,
            sampling_interval_ms=sampling_interval_ms,
            timeseries_stride=timeseries_stride,
            profile_power=effective_profile_power,
            power_sampling_interval_ms=effective_power_sampling_interval_ms,
            ui_update_every=ui_update_every,
            query_ids_path=(str(query_ids_path) if query_ids_path else str(qid_path)),
        )
        save_json(run_path / "run_manifest.json", manifest)

        setup_progress.update(setup_task_id, description="QA bootstrap: chunking corpus + qrels")
        chunks, doc_to_chunks = chunk_documents(
            docs,
            chunk_size_words=int(cfg["chunking"]["chunk_size_words"]),
            chunk_overlap_words=int(cfg["chunking"]["chunk_overlap_words"]),
            min_chunk_words=int(cfg["chunking"]["min_chunk_words"]),
        )

        doc_qrels = build_qrels_from_qa_examples(examples)
        chunk_qrels = map_doc_qrels_to_chunk_qrels(doc_qrels, doc_to_chunks)

        setup_progress.update(setup_task_id, description="QA bootstrap: loading retrievers + LLM")
        pipeline = RAGPipeline(cfg=cfg, items=chunks_to_items(chunks), enable_llm=True)
        setup_progress.update(setup_task_id, description="QA bootstrap: ready")

    runs: dict[str, list[str]] = {}
    em_values: list[float] = []
    f1_values: list[float] = []
    answerable_values: list[float] = []
    abstained_count = 0
    retrieval_latency_total: list[float] = []
    t_bm25_vals: list[float] = []
    t_qembed_vals: list[float] = []
    t_vsearch_vals: list[float] = []
    t_merge_vals: list[float] = []
    t_rerank_vals: list[float] = []
    t_total_vals: list[float] = []
    ttft_vals: list[float] = []
    t_prefill_vals: list[float] = []
    t_decode_vals: list[float] = []
    context_tokens_vals: list[float] = []
    output_tokens_vals: list[float] = []
    total_tokens_vals: list[float] = []
    tps_decode_vals: list[float] = []
    rss_peak_vals: list[float] = []
    rss_mean_vals: list[float] = []
    cpu_peak_vals: list[float] = []
    cpu_mean_vals: list[float] = []
    power_peak_vals: list[float] = []
    power_mean_vals: list[float] = []
    power_samples_total = 0
    power_status_counts: dict[str, int] = {}
    overlap_count_vals: list[float] = []
    overlap_ratio_vals: list[float] = []
    acb_k_eff_vals: list[float] = []
    acb_context_used_vals: list[float] = []
    acb_margin_vals: list[float] = []
    acb_agreement_vals: list[float] = []
    acb_low_count = 0
    acb_high_count = 0
    acb_medium_count = 0
    acb_fallback_high_count = 0
    post_support_doc_in_ctx2_vals: list[float] = []
    post_pair_in_ctx_vals: list[float] = []
    post_redundancy_avg_vals: list[float] = []
    post_redundancy_max_vals: list[float] = []
    post_mmr_lambda_max_vals: list[float] = []
    sp3_last: dict[str, Any] = _empty_sp3(cfg)
    support_doc_recall_vals_by_k: dict[int, list[float]] = {int(k): [] for k in ks}
    pair_recall_vals_by_k: dict[int, list[float]] = {int(k): [] for k in ks}
    pair_recall_eligible_queries = 0
    ui_every = max(1, int(ui_update_every))
    total_queries = len(examples)
    failures_count = 0
    valid_answer_count = 0
    pending_advance = 0

    bar_progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
    )
    status_table = _qa_status_table(
        em_values=em_values,
        f1_values=f1_values,
        valid_answer_count=valid_answer_count,
        failure_count=failures_count,
        processed=0,
        t_total_vals=t_total_vals,
        ttft_vals=ttft_vals,
        tps_decode_vals=tps_decode_vals,
        rss_peak_vals=rss_peak_vals,
        cpu_mean_vals=cpu_mean_vals,
        power_mean_vals=power_mean_vals,
        power_peak_vals=power_peak_vals,
    )

    with Live(Group(bar_progress, status_table), refresh_per_second=4) as live:
        bar_progress.start()
        try:
            task_id = bar_progress.add_task(f"QA {dataset}:{tier}", total=total_queries)
            live.update(Group(bar_progress, status_table))

            for idx, ex in enumerate(examples):
                sampler = QueryResourceSampler(
                    sampling_interval_ms=sampling_interval_ms,
                    include_timeseries=profile_timeseries,
                    timeseries_stride=timeseries_stride,
                    profile_power=effective_profile_power,
                    power_sampling_interval_ms=effective_power_sampling_interval_ms,
                )
                out = None
                error_msg = ""
                sampler.start()
                try:
                    out = pipeline.answer(ex.question, cfg)
                except Exception as exc:  # pragma: no cover - defensive path
                    error_msg = f"{exc.__class__.__name__}: {exc}"
                resources = sampler.stop()

                failed = out is None
                if failed:
                    failures_count += 1
                    pred = ""
                    retrieved = []
                    retrieved_doc_ids = []
                    retrieval_stages = _empty_retrieval_stages(cfg)
                    context_budgeting = _empty_context_budgeting(cfg)
                    post_context = _empty_post_context()
                    sp3 = _empty_sp3(cfg)
                    if error_msg:
                        retrieval_stages["error"] = error_msg
                    latency = _empty_latency(include_llm=True)
                    token_stats = _empty_token_stats()
                else:
                    pred = out.answer or ""
                    retrieved = out.retrieved
                    retrieved_doc_ids = unique_doc_ids(retrieved)
                    retrieval_stages = dict(out.retrieval_stages)
                    context_budgeting = dict(out.context_budgeting)
                    post_context = dict(out.post_context or _empty_post_context())
                    sp3 = dict(out.sp3)
                    latency = dict(out.latency_ms)
                    token_stats = dict(out.token_stats)

                qas = qa_scores(pred, ex.answers)
                em_values.append(float(qas["em"]))
                f1_values.append(float(qas["f1"]))
                abstained = pred.strip().lower() in {"", "non lo so", "i don't know", "idk"}
                if abstained:
                    abstained_count += 1
                if not abstained and not failed:
                    answerable_values.append(float(qas["em"]))
                    valid_answer_count += 1

                runs[ex.qid] = [h.item_id for h in retrieved]
                rel_for_q = chunk_qrels.get(ex.qid, {})
                rel_metrics_q, _ = evaluate_retrieval({ex.qid: runs[ex.qid]}, {ex.qid: rel_for_q}, ks=ks)
                support_doc_metrics = _support_doc_recall_at_ks(retrieved_doc_ids, ex.gold_doc_ids, ks)
                pair_recall_metrics, pair_recall_eligible = _pair_recall_at_ks(
                    retrieved_doc_ids, ex.gold_doc_ids, ks
                )
                for k in ks:
                    support_key = f"support_doc_recall@{k}"
                    support_doc_recall_vals_by_k[int(k)].append(float(support_doc_metrics.get(support_key, 0.0)))
                if pair_recall_eligible:
                    pair_recall_eligible_queries += 1
                    for k in ks:
                        pair_key = f"pair_recall@{k}"
                        pair_recall_vals_by_k[int(k)].append(float(pair_recall_metrics.get(pair_key, 0.0)))

                context_doc_ids_used = [
                    str(doc_id)
                    for doc_id in post_context.get("context_doc_ids_used", [])
                    if str(doc_id).strip()
                ]
                support_ctx2 = _support_doc_in_context_at_2(context_doc_ids_used, ex.gold_doc_ids)
                pair_ctxk = _pair_in_context_at_k(context_doc_ids_used, ex.gold_doc_ids)
                post_context["support_doc_in_context_at_2"] = support_ctx2
                post_context["pair_in_context_at_k"] = pair_ctxk
                if support_ctx2 is not None:
                    post_support_doc_in_ctx2_vals.append(float(support_ctx2))
                if pair_ctxk is not None:
                    post_pair_in_ctx_vals.append(float(pair_ctxk))
                post_redundancy_avg_vals.append(float(post_context.get("redundancy_avg_similarity", 0.0) or 0.0))
                post_redundancy_max_vals.append(float(post_context.get("redundancy_max_similarity", 0.0) or 0.0))
                post_mmr_lambda_max_vals.append(float(post_context.get("mmr_lambda_max_used", 0.0) or 0.0))

                retrieval_latency_total.append(float(latency.get("t_retrieval_total_ms", 0.0)))
                t_bm25_vals.append(float(latency.get("t_bm25_search_ms", 0.0)))
                t_qembed_vals.append(float(latency.get("t_query_embed_ms", 0.0)))
                t_vsearch_vals.append(float(latency.get("t_vector_search_ms", 0.0)))
                t_merge_vals.append(float(latency.get("t_merge_hybrid_ms", 0.0)))
                t_rerank_vals.append(float(latency.get("t_rerank_total_ms", 0.0)))
                t_total_vals.append(float(latency.get("t_total_ms", 0.0)))
                ttft_vals.append(float(latency.get("ttft_ms", 0.0)))
                t_prefill_vals.append(float(latency.get("t_prefill_ms", 0.0)))
                t_decode_vals.append(float(latency.get("t_decode_total_ms", 0.0)))

                context_tokens_vals.append(float(token_stats.get("context_tokens", 0.0)))
                output_tokens_vals.append(float(token_stats.get("output_tokens", 0.0)))
                total_tokens_vals.append(float(token_stats.get("total_tokens", 0.0)))
                tps_decode_vals.append(float(token_stats.get("tokens_per_second_decode", 0.0)))

                overlap_count = float(retrieval_stages.get("overlap_count", 0.0))
                overlap_ratio = float(retrieval_stages.get("overlap_ratio", 0.0))
                overlap_count_vals.append(overlap_count)
                overlap_ratio_vals.append(overlap_ratio)
                acb_k_eff_vals.append(float(context_budgeting.get("k_eff", 0.0) or 0.0))
                acb_context_used_vals.append(float(context_budgeting.get("context_tokens_used", 0.0) or 0.0))
                margin_val = context_budgeting.get("margin_value", None)
                agreement_val = context_budgeting.get("agreement_value", None)
                if margin_val is not None:
                    acb_margin_vals.append(float(margin_val))
                if agreement_val is not None:
                    acb_agreement_vals.append(float(agreement_val))
                branch = str(context_budgeting.get("policy_branch", "")).lower()
                if branch == "low":
                    acb_low_count += 1
                elif branch == "high":
                    acb_high_count += 1
                elif branch == "medium":
                    acb_medium_count += 1
                if bool(context_budgeting.get("fallback_to_high", False)):
                    acb_fallback_high_count += 1

                rss_peak_vals.append(float(resources.rss_peak_bytes))
                rss_mean_vals.append(float(resources.rss_mean_bytes))
                cpu_peak_vals.append(float(resources.cpu_peak_pct))
                cpu_mean_vals.append(float(resources.cpu_mean_pct))
                if resources.power_peak_watts is not None:
                    power_peak_vals.append(float(resources.power_peak_watts))
                if resources.power_mean_watts is not None:
                    power_mean_vals.append(float(resources.power_mean_watts))
                power_samples_total += int(resources.power_samples)
                power_key = str(resources.power_status or "unknown")
                power_status_counts[power_key] = int(power_status_counts.get(power_key, 0) + 1)

                timeseries_ref = ""
                points_written = 0
                if profile_timeseries:
                    timeseries_ref, points_written = _write_timeseries(
                        out_dir=timeseries_dir,
                        qid=ex.qid,
                        rss_ts=resources.rss_timeseries,
                        cpu_ts=resources.cpu_timeseries,
                    )

                row = {
                    "run_id": run_id,
                    "config_id": config_fingerprint(cfg),
                    "dataset": dataset,
                    "split": split,
                    "tier": tier,
                    "query_index": idx,
                    "qid": ex.qid,
                    "question": ex.question,
                    "answer_refs": ex.answers,
                    "prediction": pred,
                    "abstained": abstained,
                    "failed": failed,
                    "failure_reason": error_msg,
                    "retrieved_ids": [r.item_id for r in retrieved],
                    "retrieved_doc_ids": retrieved_doc_ids,
                    "retrieval_metrics_per_query": rel_metrics_q,
                    "answer_metrics_per_query": {
                        "em": float(qas["em"]),
                        "f1": float(qas["f1"]),
                        "rouge_l": 0.0,
                        "unsupported_heuristic": False,
                        "support_doc_metrics": support_doc_metrics,
                        "pair_recall_metrics": pair_recall_metrics,
                        "pair_recall_eligible": bool(pair_recall_eligible),
                        "support_fact_overlap": 0.0,
                    },
                    "retrieval_stages": retrieval_stages,
                    "latency_ms": latency,
                    "tokens": token_stats,
                    "context_budgeting": context_budgeting,
                    "post_context": post_context,
                    "sp3": sp3,
                    "resources": {
                        "rss_peak_bytes": int(resources.rss_peak_bytes),
                        "rss_mean_bytes": float(resources.rss_mean_bytes),
                        "cpu_peak_pct": float(resources.cpu_peak_pct),
                        "cpu_mean_pct": float(resources.cpu_mean_pct),
                        "power_peak_watts": (
                            float(resources.power_peak_watts)
                            if resources.power_peak_watts is not None
                            else None
                        ),
                        "power_mean_watts": (
                            float(resources.power_mean_watts)
                            if resources.power_mean_watts is not None
                            else None
                        ),
                        "power_samples": int(resources.power_samples),
                        "power_status": str(resources.power_status),
                        "power_backend": resources.power_backend,
                        "power_sampling_interval_ms": (
                            int(effective_power_sampling_interval_ms) if effective_profile_power else None
                        ),
                        "timeseries_ref": timeseries_ref,
                        "timeseries_points_written": int(points_written),
                        "sampling_interval_ms": int(sampling_interval_ms),
                        "timeseries_stride": int(timeseries_stride),
                    },
                }
                _write_jsonl(per_query_path, row)
                sp3_last = dict(sp3)

                pending_advance += 1
                processed = idx + 1
                if pending_advance >= ui_every or processed == total_queries:
                    bar_progress.update(task_id, advance=pending_advance)
                    status_table = _qa_status_table(
                        em_values=em_values,
                        f1_values=f1_values,
                        valid_answer_count=valid_answer_count,
                        failure_count=failures_count,
                        processed=processed,
                        t_total_vals=t_total_vals,
                        ttft_vals=ttft_vals,
                        tps_decode_vals=tps_decode_vals,
                        rss_peak_vals=rss_peak_vals,
                        cpu_mean_vals=cpu_mean_vals,
                        power_mean_vals=power_mean_vals,
                        power_peak_vals=power_peak_vals,
                    )
                    live.update(Group(bar_progress, status_table))
                    pending_advance = 0
        finally:
            bar_progress.stop()

    ret_metrics, ret_per_query = evaluate_retrieval(runs, chunk_qrels, ks=ks)
    cb_summary_resolved = resolve_context_budgeting_config(cfg.get("context_budgeting", {}))
    summary = {
        "task": "qa_rag",
        "run_id": run_id,
        "config_id": config_fingerprint(cfg),
        "dataset": dataset,
        "split": split,
        "tier": tier,
        "num_docs": len(docs),
        "num_chunks": len(chunks),
        "num_queries": len(examples),
        "retriever_mode": cfg["retrieval"].get("retriever_mode", "hybrid"),
        "fusion_method": cfg["retrieval"].get("fusion_method", "RRF"),
        "rrf_k": int(cfg["retrieval"].get("rrf_k", 60)),
        "weighted_alpha": (
            float(cfg["retrieval"].get("weighted_alpha", cfg["retrieval"].get("hybrid_alpha", 0.5)))
            if str(cfg["retrieval"].get("fusion_method", "RRF")).upper() == "WEIGHTED_SUM"
            else None
        ),
        "agreement_bonus_enabled": bool(cfg["retrieval"].get("agreement_bonus_enabled", False)),
        "agreement_bonus": float(cfg["retrieval"].get("agreement_bonus", 0.0)),
        "sp3": sp3_last,
        "retrieval_metrics": ret_metrics,
        "retrieval_per_query": ret_per_query,
        "supporting_docs": {
            "support_doc_recall": {
                f"support_doc_recall@{k}": _mean(vs) for k, vs in sorted(support_doc_recall_vals_by_k.items())
            },
            "pair_recall": {
                f"pair_recall@{k}": (_mean(vs) if vs else None) for k, vs in sorted(pair_recall_vals_by_k.items())
            },
            "pair_recall_eligible_queries": int(pair_recall_eligible_queries),
        },
        "generation": {
            "EM": float(mean(em_values)) if em_values else 0.0,
            "F1": float(mean(f1_values)) if f1_values else 0.0,
            "ROUGE_L": 0.0,
            "valid_answer_rate": float(valid_answer_count / len(em_values)) if em_values else 0.0,
            "failure_rate": float(failures_count / len(em_values)) if em_values else 0.0,
            "abstention_rate": float(abstained_count / len(em_values)) if em_values else 0.0,
            "accuracy_when_answered": float(mean(answerable_values)) if answerable_values else 0.0,
            "em_values": em_values,
            "f1_values": f1_values,
        },
        "latency_ms": {
            "t_retrieval_total_ms": _aggregate_summary_stats(retrieval_latency_total),
            "t_bm25_search_ms": _aggregate_summary_stats(t_bm25_vals),
            "t_query_embed_ms": _aggregate_summary_stats(t_qembed_vals),
            "t_vector_search_ms": _aggregate_summary_stats(t_vsearch_vals),
            "t_merge_hybrid_ms": _aggregate_summary_stats(t_merge_vals),
            "t_rerank_total_ms": _aggregate_summary_stats(t_rerank_vals),
            "t_total_ms": _aggregate_summary_stats(t_total_vals),
            "ttft_ms": _aggregate_summary_stats(ttft_vals),
            "t_prefill_ms": _aggregate_summary_stats(t_prefill_vals),
            "t_decode_total_ms": _aggregate_summary_stats(t_decode_vals),
        },
        "tokens": {
            "context_tokens": _aggregate_summary_stats(context_tokens_vals),
            "output_tokens": _aggregate_summary_stats(output_tokens_vals),
            "total_tokens": _aggregate_summary_stats(total_tokens_vals),
            "tokens_per_second_decode": _aggregate_summary_stats(tps_decode_vals),
        },
        "resources": {
            "rss_peak_bytes": _aggregate_summary_stats(rss_peak_vals),
            "rss_mean_bytes": _aggregate_summary_stats(rss_mean_vals),
            "cpu_peak_pct": _aggregate_summary_stats(cpu_peak_vals),
            "cpu_mean_pct": _aggregate_summary_stats(cpu_mean_vals),
            "power_peak_watts": _aggregate_summary_stats(power_peak_vals),
            "power_mean_watts": _aggregate_summary_stats(power_mean_vals),
            "power_samples_total": int(power_samples_total),
            "power_available_queries": int(len(power_mean_vals)),
            "power_status_counts": dict(power_status_counts),
            "power_backend": ("powermetrics" if effective_profile_power else None),
            "profile_power": bool(effective_profile_power),
        },
        "overlap": {
            "overlap_count": _aggregate_summary_stats(overlap_count_vals),
            "overlap_ratio": _aggregate_summary_stats(overlap_ratio_vals),
        },
        "context_budgeting": {
            "enabled": bool(cb_summary_resolved.get("enabled", cfg.get("context_budgeting", {}).get("enabled", False))),
            "strategy": str(cb_summary_resolved.get("strategy", cfg.get("context_budgeting", {}).get("strategy", "v1"))).strip().lower(),
            "k_eff": _aggregate_summary_stats(acb_k_eff_vals),
            "context_tokens_used": _aggregate_summary_stats(acb_context_used_vals),
            "pct_low_branch": float(acb_low_count / max(1, len(examples))),
            "pct_medium_branch": float(acb_medium_count / max(1, len(examples))),
            "pct_high_branch": float(acb_high_count / max(1, len(examples))),
            "pct_fallback_to_high": float(acb_fallback_high_count / max(1, len(examples))),
            "margin": _aggregate_summary_stats(acb_margin_vals),
            "agreement": _aggregate_summary_stats(acb_agreement_vals),
            "aliases_used": dict(cb_summary_resolved.get("aliases_used", {})),
            "unknown_keys": list(cb_summary_resolved.get("unknown_keys", [])),
            "v2_resolved_params": (
                {
                    "keep_full_count": int(cb_summary_resolved.get("keep_full_count", 2)),
                    "mmr_lambda": float(cb_summary_resolved.get("mmr_lambda", 0.30)),
                    "min_low_savings_ratio": float(cb_summary_resolved.get("min_low_savings_ratio", 0.15)),
                    "k_eff_floor": int(cb_summary_resolved.get("k_eff_floor", 5)),
                    "snippet_from_rank": int(cb_summary_resolved.get("snippet_from_rank", 3)),
                    "low_margin_multiplier": float(cb_summary_resolved.get("low_margin_multiplier", 5.0)),
                    "low_agreement_multiplier": float(cb_summary_resolved.get("low_agreement_multiplier", 1.0)),
                    "medium_branch_enabled": bool(cb_summary_resolved.get("medium_branch_enabled", True)),
                    "medium_budget_tokens": int(cb_summary_resolved.get("medium_budget_tokens", 900)),
                    "medium_k_eff": int(cb_summary_resolved.get("medium_k_eff", 8)),
                    "top_doc_saliency_tokens": int(cb_summary_resolved.get("top_doc_saliency_tokens", 192)),
                    "saliency_entity_weight": float(cb_summary_resolved.get("saliency_entity_weight", 2.0)),
                    "dynamic_mmr_enabled": bool(cb_summary_resolved.get("dynamic_mmr_enabled", True)),
                    "dynamic_mmr_threshold": float(cb_summary_resolved.get("dynamic_mmr_threshold", 0.45)),
                    "dynamic_mmr_boost": float(cb_summary_resolved.get("dynamic_mmr_boost", 0.60)),
                    "dynamic_mmr_cap": float(cb_summary_resolved.get("dynamic_mmr_cap", 1.20)),
                }
                if str(cb_summary_resolved.get("strategy", "v1")).strip().lower() == "v2_evidence_first"
                else {}
            ),
        },
        "post_context": {
            "support_doc_in_context_at_2_mean": (
                _mean(post_support_doc_in_ctx2_vals) if post_support_doc_in_ctx2_vals else None
            ),
            "pair_in_context_at_k_mean": (_mean(post_pair_in_ctx_vals) if post_pair_in_ctx_vals else None),
            "redundancy_avg_similarity": _aggregate_summary_stats(post_redundancy_avg_vals),
            "redundancy_max_similarity": _aggregate_summary_stats(post_redundancy_max_vals),
            "mmr_lambda_max_used": _aggregate_summary_stats(post_mmr_lambda_max_vals),
            "pct_fallback_to_high": float(acb_fallback_high_count / max(1, len(examples))),
        },
        "artifacts": {
            "run_manifest": str(run_path / "run_manifest.json"),
            "per_query_jsonl": str(per_query_path),
            "timeseries_dir": str(timeseries_dir) if profile_timeseries else "",
        },
    }
    save_json(run_path / "summary.json", summary)
    return summary


def run_beir_retrieval_benchmark(
    cfg: dict[str, Any],
    beir_datasets: list[str],
    tier: str,
    output_dir: str | Path,
    run_id: str,
    seed: int,
    max_queries_override: int | None = None,
    retrieval_ks: list[int] | None = None,
    profile_timeseries: bool = False,
    sampling_interval_ms: int = 200,
    timeseries_stride: int = 5,
    profile_power: bool | None = None,
    power_sampling_interval_ms: int | None = None,
    ui_update_every: int = 5,
) -> dict[str, Any]:
    ks = sorted(set([int(k) for k in (retrieval_ks or DEFAULT_RETRIEVAL_KS)]))
    max_queries = resolve_query_budget("beir", tier, max_queries_override)
    profiling_cfg = cfg.get("profiling", {})
    effective_profile_power = (
        bool(profile_power) if profile_power is not None else bool(profiling_cfg.get("profile_power", False))
    )
    effective_power_sampling_interval_ms = (
        int(power_sampling_interval_ms)
        if power_sampling_interval_ms is not None
        else int(profiling_cfg.get("power_sampling_interval_ms", 1000))
    )
    run_path = Path(output_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    per_query_path = run_path / "per_query.jsonl"
    if per_query_path.exists():
        per_query_path.unlink()
    timeseries_dir = run_path / "timeseries"

    manifest = _make_run_manifest(
        run_id=run_id,
        cfg=cfg,
        dataset="beir",
        split="test",
        tier=tier,
        seed=seed,
        num_queries=max_queries,
        profile_timeseries=profile_timeseries,
        sampling_interval_ms=sampling_interval_ms,
        timeseries_stride=timeseries_stride,
        profile_power=effective_profile_power,
        power_sampling_interval_ms=effective_power_sampling_interval_ms,
        ui_update_every=ui_update_every,
        query_ids_path="",
    )
    manifest["beir_datasets"] = beir_datasets
    save_json(run_path / "run_manifest.json", manifest)

    outputs: dict[str, Any] = {
        "task": "beir_retrieval",
        "run_id": run_id,
        "config_id": config_fingerprint(cfg),
        "tier": tier,
        "retriever_mode": cfg["retrieval"].get("retriever_mode", "hybrid"),
        "fusion_method": cfg["retrieval"].get("fusion_method", "RRF"),
        "rrf_k": int(cfg["retrieval"].get("rrf_k", 60)),
        "weighted_alpha": (
            float(cfg["retrieval"].get("weighted_alpha", cfg["retrieval"].get("hybrid_alpha", 0.5)))
            if str(cfg["retrieval"].get("fusion_method", "RRF")).upper() == "WEIGHTED_SUM"
            else None
        ),
        "sp3": _empty_sp3(cfg),
        "context_budgeting": _empty_context_budgeting(cfg),
        "datasets": {},
        "macro_avg": {},
        "artifacts": {
            "run_manifest": str(run_path / "run_manifest.json"),
            "per_query_jsonl": str(per_query_path),
            "timeseries_dir": str(timeseries_dir) if profile_timeseries else "",
        },
    }

    dataset_scores: list[dict[str, float]] = []
    overlap_ratios: list[float] = []
    overlap_counts: list[float] = []
    t_ret_vals: list[float] = []
    t_bm25_vals: list[float] = []
    t_qembed_vals: list[float] = []
    t_vsearch_vals: list[float] = []
    t_merge_vals: list[float] = []
    rss_peak_vals: list[float] = []
    cpu_mean_vals: list[float] = []
    power_peak_vals: list[float] = []
    power_mean_vals: list[float] = []
    power_samples_total = 0
    power_status_counts: dict[str, int] = {}
    failure_count_total = 0
    ui_every = max(1, int(ui_update_every))
    for ds_name in beir_datasets:
        docs, queries, qrels = load_beir_dataset(
            dataset_name=ds_name,
            max_queries=max_queries,
            seed=seed,
        )
        items = docs_to_items(docs)
        pipeline = RAGPipeline(cfg=cfg, items=items, enable_llm=False)

        runs: dict[str, list[str]] = {}
        per_q: dict[str, dict[str, float]] = {}
        ds_t_ret: list[float] = []
        ds_recall10_vals: list[float] = []
        ds_ndcg10_vals: list[float] = []
        ds_failures = 0
        pending_advance = 0
        total_ds_queries = len(queries)

        bar_progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=True,
        )
        status_table = _beir_status_table(
            recall10_vals=ds_recall10_vals,
            ndcg10_vals=ds_ndcg10_vals,
            failure_count=ds_failures,
            processed=0,
            t_total_vals=ds_t_ret,
            rss_peak_vals=rss_peak_vals,
            cpu_mean_vals=cpu_mean_vals,
            power_mean_vals=power_mean_vals,
            power_peak_vals=power_peak_vals,
        )

        with Live(Group(bar_progress, status_table), refresh_per_second=4) as live:
            bar_progress.start()
            try:
                task_id = bar_progress.add_task(f"BEIR {ds_name}:{tier}", total=total_ds_queries)
                live.update(Group(bar_progress, status_table))

                for idx, (qid, question) in enumerate(queries.items()):
                    sampler = QueryResourceSampler(
                        sampling_interval_ms=sampling_interval_ms,
                        include_timeseries=profile_timeseries,
                        timeseries_stride=timeseries_stride,
                        profile_power=effective_profile_power,
                        power_sampling_interval_ms=effective_power_sampling_interval_ms,
                    )
                    hits = []
                    trace = _empty_retrieval_stages(cfg)
                    error_msg = ""
                    sampler.start()
                    try:
                        hits, trace, _ = pipeline.retrieve_with_trace(question, cfg)
                    except Exception as exc:  # pragma: no cover - defensive path
                        error_msg = f"{exc.__class__.__name__}: {exc}"
                    resources = sampler.stop()

                    failed = bool(error_msg)
                    if failed:
                        ds_failures += 1
                        failure_count_total += 1
                        trace = _empty_retrieval_stages(cfg)
                        trace["error"] = error_msg

                    ranked = unique_doc_ids(hits)
                    runs[qid] = ranked
                    q_metrics, _ = evaluate_retrieval({qid: ranked}, {qid: qrels.get(qid, {})}, ks=ks)
                    per_q[qid] = q_metrics

                    ds_recall10_vals.append(float(q_metrics.get("Recall@10", 0.0)))
                    ds_ndcg10_vals.append(float(q_metrics.get("nDCG@10", 0.0)))

                    t_ret = float(trace.get("t_retrieval_total_ms", 0.0))
                    ds_t_ret.append(t_ret)
                    t_ret_vals.append(t_ret)
                    t_bm25_vals.append(float(trace.get("bm25", {}).get("t_bm25_search_ms", 0.0)))
                    t_qembed_vals.append(float(trace.get("dense", {}).get("t_query_embed_ms", 0.0)))
                    t_vsearch_vals.append(float(trace.get("dense", {}).get("t_vector_search_ms", 0.0)))
                    t_merge_vals.append(float(trace.get("fusion", {}).get("t_merge_hybrid_ms", 0.0)))
                    overlap_counts.append(float(trace.get("overlap_count", 0.0)))
                    overlap_ratios.append(float(trace.get("overlap_ratio", 0.0)))
                    rss_peak_vals.append(float(resources.rss_peak_bytes))
                    cpu_mean_vals.append(float(resources.cpu_mean_pct))
                    if resources.power_peak_watts is not None:
                        power_peak_vals.append(float(resources.power_peak_watts))
                    if resources.power_mean_watts is not None:
                        power_mean_vals.append(float(resources.power_mean_watts))
                    power_samples_total += int(resources.power_samples)
                    power_key = str(resources.power_status or "unknown")
                    power_status_counts[power_key] = int(power_status_counts.get(power_key, 0) + 1)

                    timeseries_ref = ""
                    points_written = 0
                    if profile_timeseries:
                        timeseries_ref, points_written = _write_timeseries(
                            out_dir=timeseries_dir / ds_name,
                            qid=qid,
                            rss_ts=resources.rss_timeseries,
                            cpu_ts=resources.cpu_timeseries,
                        )

                    row = {
                        "run_id": run_id,
                        "config_id": config_fingerprint(cfg),
                        "dataset": f"beir:{ds_name}",
                        "split": "test",
                        "tier": tier,
                        "query_index": idx,
                        "qid": qid,
                        "question": question,
                        "answer_refs": [],
                        "prediction": None,
                        "abstained": False,
                        "failed": failed,
                        "failure_reason": error_msg,
                        "retrieved_ids": ranked,
                        "retrieved_doc_ids": ranked,
                        "retrieval_metrics_per_query": q_metrics,
                        "answer_metrics_per_query": _blank_generation(),
                        "retrieval_stages": trace,
                        "latency_ms": {
                            "t_retrieval_total_ms": t_ret,
                            "t_bm25_search_ms": float(trace.get("bm25", {}).get("t_bm25_search_ms", 0.0)),
                            "t_query_embed_ms": float(trace.get("dense", {}).get("t_query_embed_ms", 0.0)),
                            "t_vector_search_ms": float(trace.get("dense", {}).get("t_vector_search_ms", 0.0)),
                            "t_merge_hybrid_ms": float(trace.get("fusion", {}).get("t_merge_hybrid_ms", 0.0)),
                            "t_rerank_total_ms": 0.0,
                            "ttft_ms": 0.0,
                            "t_total_ms": t_ret,
                        },
                        "tokens": _empty_token_stats(),
                        "context_budgeting": _empty_context_budgeting(cfg),
                        "sp3": _empty_sp3(cfg),
                        "resources": {
                            "rss_peak_bytes": int(resources.rss_peak_bytes),
                            "rss_mean_bytes": float(resources.rss_mean_bytes),
                            "cpu_peak_pct": float(resources.cpu_peak_pct),
                            "cpu_mean_pct": float(resources.cpu_mean_pct),
                            "power_peak_watts": (
                                float(resources.power_peak_watts)
                                if resources.power_peak_watts is not None
                                else None
                            ),
                            "power_mean_watts": (
                                float(resources.power_mean_watts)
                                if resources.power_mean_watts is not None
                                else None
                            ),
                            "power_samples": int(resources.power_samples),
                            "power_status": str(resources.power_status),
                            "power_backend": resources.power_backend,
                            "power_sampling_interval_ms": (
                                int(effective_power_sampling_interval_ms) if effective_profile_power else None
                            ),
                            "timeseries_ref": timeseries_ref,
                            "timeseries_points_written": int(points_written),
                            "sampling_interval_ms": int(sampling_interval_ms),
                            "timeseries_stride": int(timeseries_stride),
                        },
                    }
                    _write_jsonl(per_query_path, row)

                    pending_advance += 1
                    processed = idx + 1
                    if pending_advance >= ui_every or processed == total_ds_queries:
                        bar_progress.update(task_id, advance=pending_advance)
                        status_table = _beir_status_table(
                            recall10_vals=ds_recall10_vals,
                            ndcg10_vals=ds_ndcg10_vals,
                            failure_count=ds_failures,
                            processed=processed,
                            t_total_vals=ds_t_ret,
                            rss_peak_vals=rss_peak_vals,
                            cpu_mean_vals=cpu_mean_vals,
                            power_mean_vals=power_mean_vals,
                            power_peak_vals=power_peak_vals,
                        )
                        live.update(Group(bar_progress, status_table))
                        pending_advance = 0
            finally:
                bar_progress.stop()

        metrics, per_query = evaluate_retrieval(runs, qrels, ks=ks)
        outputs["datasets"][ds_name] = {
            "num_docs": len(docs),
            "num_queries": len(queries),
            "metrics": metrics,
            "per_query": per_query,
            "latency_ms": {"t_retrieval_total_ms": _aggregate_summary_stats(ds_t_ret)},
        }
        dataset_scores.append(metrics)

    if dataset_scores:
        keys = sorted({k for d in dataset_scores for k in d})
        outputs["macro_avg"] = {
            k: float(sum(d.get(k, 0.0) for d in dataset_scores) / len(dataset_scores)) for k in keys
        }

    outputs["latency_ms"] = {
        "t_retrieval_total_ms": _aggregate_summary_stats(t_ret_vals),
        "t_bm25_search_ms": _aggregate_summary_stats(t_bm25_vals),
        "t_query_embed_ms": _aggregate_summary_stats(t_qembed_vals),
        "t_vector_search_ms": _aggregate_summary_stats(t_vsearch_vals),
        "t_merge_hybrid_ms": _aggregate_summary_stats(t_merge_vals),
    }
    outputs["resources"] = {
        "rss_peak_bytes": _aggregate_summary_stats(rss_peak_vals),
        "cpu_mean_pct": _aggregate_summary_stats(cpu_mean_vals),
        "power_peak_watts": _aggregate_summary_stats(power_peak_vals),
        "power_mean_watts": _aggregate_summary_stats(power_mean_vals),
        "power_samples_total": int(power_samples_total),
        "power_available_queries": int(len(power_mean_vals)),
        "power_status_counts": dict(power_status_counts),
        "power_backend": ("powermetrics" if effective_profile_power else None),
        "profile_power": bool(effective_profile_power),
    }
    outputs["failure_rate"] = float(failure_count_total / max(1, len(t_ret_vals)))
    outputs["overlap"] = {
        "overlap_count": _aggregate_summary_stats(overlap_counts),
        "overlap_ratio": _aggregate_summary_stats(overlap_ratios),
    }

    save_json(run_path / "summary.json", outputs)
    return outputs


def write_summary_markdown(path: str | Path, summary: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    def _triplet(value: Any) -> tuple[float, float, float]:
        if isinstance(value, dict):
            return (
                float(value.get("mean", 0.0)),
                float(value.get("p50", 0.0)),
                float(value.get("p95", 0.0)),
            )
        v = float(value or 0.0)
        return (v, v, v)

    lines: list[str] = []
    lines.append("# Benchmark Summary")
    lines.append("")
    lines.append(f"- run_id: `{summary.get('run_id', '')}`")
    lines.append(f"- config_id: `{summary.get('config_id', '')}`")
    lines.append(f"- task: `{summary.get('task', '')}`")
    lines.append(f"- dataset: `{summary.get('dataset', '')}`")
    lines.append(f"- tier: `{summary.get('tier', '')}`")
    lines.append(f"- retriever_mode: `{summary.get('retriever_mode', '')}`")
    lines.append(f"- fusion_method: `{summary.get('fusion_method', '')}`")
    lines.append(f"- rrf_k: `{summary.get('rrf_k', '')}`")
    lines.append(f"- weighted_alpha: `{summary.get('weighted_alpha', None)}`")
    sp3 = summary.get("sp3", {})
    lines.append(f"- sp3_profile: `{sp3.get('profile', '')}`")
    lines.append(f"- threads_decode: `{sp3.get('threads_decode', '')}`")
    lines.append(f"- threads_batch: `{sp3.get('threads_batch', '')}`")
    lines.append("")
    if "retrieval_metrics" in summary:
        lines.append("## Retrieval")
        lines.append("")
        for k, v in sorted(summary["retrieval_metrics"].items()):
            lines.append(f"- {k}: {v:.4f}")
        lines.append("")
    if "supporting_docs" in summary:
        supp = summary.get("supporting_docs", {})
        lines.append("## Supporting Docs")
        lines.append("")
        lines.append(f"- pair_recall_eligible_queries: {int(supp.get('pair_recall_eligible_queries', 0))}")
        for k, v in sorted(supp.get("support_doc_recall", {}).items()):
            lines.append(f"- {k}: {float(v):.4f}")
        for k, v in sorted(supp.get("pair_recall", {}).items()):
            if v is None:
                lines.append(f"- {k}: n/a")
            else:
                lines.append(f"- {k}: {float(v):.4f}")
        lines.append("")
    if "generation" in summary:
        g = summary["generation"]
        lines.append("## Generation")
        lines.append("")
        lines.append(f"- EM: {g.get('EM', 0.0):.4f}")
        lines.append(f"- F1: {g.get('F1', 0.0):.4f}")
        lines.append(f"- abstention_rate: {g.get('abstention_rate', 0.0):.4f}")
        lines.append("")
    if "latency_ms" in summary:
        lines.append("## Latency")
        lines.append("")
        for k, v in summary["latency_ms"].items():
            if isinstance(v, dict):
                lines.append(f"- {k}: mean={v.get('mean', 0.0):.3f} p50={v.get('p50', 0.0):.3f} p95={v.get('p95', 0.0):.3f}")
        lines.append("")
    if "resources" in summary:
        lines.append("## Resources")
        lines.append("")
        for k, v in summary["resources"].items():
            if isinstance(v, dict) and {"mean", "p50", "p95"}.issubset(set(v.keys())):
                lines.append(f"- {k}: mean={v.get('mean', 0.0):.3f} p50={v.get('p50', 0.0):.3f} p95={v.get('p95', 0.0):.3f}")
            elif isinstance(v, dict):
                lines.append(f"- {k}: `{json.dumps(v, ensure_ascii=False)}`")
            else:
                lines.append(f"- {k}: {v}")
        lines.append("")
    if "context_budgeting" in summary:
        cb = summary["context_budgeting"]
        k_mean, k_p50, k_p95 = _triplet(cb.get("k_eff", 0.0))
        c_mean, c_p50, c_p95 = _triplet(cb.get("context_tokens_used", 0.0))
        lines.append("## Context Budgeting")
        lines.append("")
        lines.append(f"- enabled: {bool(cb.get('enabled', False))}")
        lines.append(f"- strategy: `{cb.get('strategy', 'v1')}`")
        lines.append(f"- k_eff mean/p50/p95: {k_mean:.3f}/{k_p50:.3f}/{k_p95:.3f}")
        lines.append(f"- context_tokens_used mean/p50/p95: {c_mean:.3f}/{c_p50:.3f}/{c_p95:.3f}")
        lines.append(
            f"- pct_low_branch/pct_medium_branch/pct_high_branch: {cb.get('pct_low_branch', 0.0):.3f}/"
            f"{cb.get('pct_medium_branch', 0.0):.3f}/{cb.get('pct_high_branch', 0.0):.3f}"
        )
        lines.append(f"- pct_fallback_to_high: {cb.get('pct_fallback_to_high', 0.0):.3f}")
        lines.append(f"- aliases_used: `{json.dumps(cb.get('aliases_used', {}), ensure_ascii=False)}`")
        lines.append(f"- unknown_keys: `{json.dumps(cb.get('unknown_keys', []), ensure_ascii=False)}`")
        lines.append(f"- v2_resolved_params: `{json.dumps(cb.get('v2_resolved_params', {}), ensure_ascii=False)}`")
        lines.append("")
    if "post_context" in summary:
        pc = summary["post_context"]
        r_avg_mean, r_avg_p50, r_avg_p95 = _triplet(pc.get("redundancy_avg_similarity", 0.0))
        r_max_mean, r_max_p50, r_max_p95 = _triplet(pc.get("redundancy_max_similarity", 0.0))
        l_mean, l_p50, l_p95 = _triplet(pc.get("mmr_lambda_max_used", 0.0))
        lines.append("## Post Context")
        lines.append("")
        lines.append(f"- support_doc_in_context_at_2_mean: {pc.get('support_doc_in_context_at_2_mean', None)}")
        lines.append(f"- pair_in_context_at_k_mean: {pc.get('pair_in_context_at_k_mean', None)}")
        lines.append(f"- pct_fallback_to_high: {pc.get('pct_fallback_to_high', 0.0):.3f}")
        lines.append(
            f"- redundancy_avg_similarity mean/p50/p95: {r_avg_mean:.3f}/{r_avg_p50:.3f}/{r_avg_p95:.3f}"
        )
        lines.append(
            f"- redundancy_max_similarity mean/p50/p95: {r_max_mean:.3f}/{r_max_p50:.3f}/{r_max_p95:.3f}"
        )
        lines.append(f"- mmr_lambda_max_used mean/p50/p95: {l_mean:.3f}/{l_p50:.3f}/{l_p95:.3f}")
        lines.append("")

    p.write_text("\n".join(lines), encoding="utf-8")
