#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_cpu.utils import save_json


TOP_QTYPES = ["what", "which", "who", "the", "are", "in", "when", "where", "how"]
STRATA_ORDER = {"hard": 0, "mid": 1, "easy": 2}


@dataclass(slots=True)
class QidMeta:
    qid: str
    question: str
    qtype: str
    answer_type: str
    avg_f1: float
    avg_pair10: float
    avg_support2: float
    all_em0: int
    difficulty: float
    stratum: str


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if math.isnan(x):
        return float(default)
    return float(x)


def _first_token(text: str) -> str:
    toks = (text or "").strip().lower().split()
    return toks[0] if toks else "other"


def _answer_type(answer_refs: Any) -> str:
    refs = answer_refs if isinstance(answer_refs, list) else []
    norm = [str(x or "").strip().lower() for x in refs]
    if norm and all(x in {"yes", "no"} for x in norm):
        return "yesno"
    return "span"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _qid_map(path: Path) -> dict[str, dict[str, Any]]:
    rows = _load_jsonl(path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = str(row.get("qid", "")).strip()
        if not qid:
            continue
        if qid not in out:
            out[qid] = row
    return out


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round((len(s) - 1) * p))))
    return float(s[idx])


def _qtype_targets(
    full_counts: Counter[str],
    full_total: int,
    pool_size: int,
    qtypes: list[str],
) -> dict[str, int]:
    total = max(1, int(full_total))
    return {
        q: int(round(pool_size * (full_counts.get(q, 0) / total)))
        for q in qtypes
    }


def _compute_ranges(targets: dict[str, int], rel_tol: float = 0.20) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for k, tgt in targets.items():
        tol = max(1, int(round(tgt * rel_tol)))
        lo = max(0, tgt - tol)
        hi = tgt + tol
        out[k] = (int(lo), int(hi))
    return out


def _collect_stats(selected: set[str], meta: dict[str, QidMeta]) -> dict[str, Any]:
    q_counter = Counter()
    a_counter = Counter()
    s_counter = Counter()
    for qid in selected:
        m = meta[qid]
        q_counter[m.qtype] += 1
        a_counter[m.answer_type] += 1
        s_counter[m.stratum] += 1
    return {
        "qtype": dict(q_counter),
        "answer_type": dict(a_counter),
        "strata": dict(s_counter),
    }


def _yesno_in_range(selected: set[str], meta: dict[str, QidMeta], target: int, tol: int) -> bool:
    yesno = sum(1 for q in selected if meta[q].answer_type == "yesno")
    return max(0, target - tol) <= yesno <= (target + tol)


def _qtypes_in_range(
    selected: set[str],
    meta: dict[str, QidMeta],
    qtype_ranges: dict[str, tuple[int, int]],
) -> tuple[bool, list[str]]:
    counts = Counter(meta[q].qtype for q in selected)
    violations: list[str] = []
    for qt, (lo, hi) in qtype_ranges.items():
        n = int(counts.get(qt, 0))
        if n < lo or n > hi:
            violations.append(f"{qt}:{n} not in [{lo},{hi}]")
    return (len(violations) == 0), violations


def _pick_with_key(cands: list[QidMeta], need: int, key_fn) -> list[QidMeta]:
    if need <= 0:
        return []
    s = sorted(cands, key=key_fn)
    return s[: min(need, len(s))]


def _select_initial(
    meta: dict[str, QidMeta],
    hard100: list[str],
    pool_size: int,
    rng: random.Random,
    fallbacks: list[str],
) -> set[str]:
    selected: set[str] = set()

    quotas = {"hard": 120, "mid": 105, "easy": 75}
    target_hard100 = 80

    hard100_hard = [q for q in hard100 if q in meta and meta[q].stratum == "hard"]
    hard100_sorted = sorted(hard100_hard, key=lambda q: (-meta[q].difficulty, q))
    forced = hard100_sorted[:target_hard100]
    if len(forced) < target_hard100:
        fallbacks.append(
            f"hard100_insufficient_in_hard: requested={target_hard100}, available={len(forced)}"
        )
    selected.update(forced)

    hard_rest = [m for m in meta.values() if m.stratum == "hard" and m.qid not in selected]
    take_hard_rest = max(
        0,
        quotas["hard"] - len([q for q in selected if meta[q].stratum == "hard"]),
    )
    hard_pick = _pick_with_key(hard_rest, take_hard_rest, key_fn=lambda m: (-m.difficulty, m.qid))
    selected.update(m.qid for m in hard_pick)
    if len(hard_pick) < take_hard_rest:
        fallbacks.append(
            f"hard_quota_underfilled_from_hard_stratum: needed={take_hard_rest}, got={len(hard_pick)}"
        )

    mid_cands = [m for m in meta.values() if m.stratum == "mid" and m.qid not in selected]
    mid_pick = _pick_with_key(mid_cands, quotas["mid"], key_fn=lambda m: (-m.difficulty, m.qid))
    selected.update(m.qid for m in mid_pick)
    if len(mid_pick) < quotas["mid"]:
        fallbacks.append(
            f"mid_quota_underfilled: needed={quotas['mid']}, got={len(mid_pick)}"
        )

    easy_cands = [m for m in meta.values() if m.stratum == "easy" and m.qid not in selected]
    easy_pick = _pick_with_key(easy_cands, quotas["easy"], key_fn=lambda m: (m.difficulty, m.qid))
    selected.update(m.qid for m in easy_pick)
    if len(easy_pick) < quotas["easy"]:
        fallbacks.append(
            f"easy_quota_underfilled: needed={quotas['easy']}, got={len(easy_pick)}"
        )

    if len(selected) < pool_size:
        missing = pool_size - len(selected)
        others = [m.qid for m in meta.values() if m.qid not in selected]
        rng.shuffle(others)
        selected.update(others[:missing])
        fallbacks.append(f"pool_padded_randomly: +{min(missing, len(others))}")
    elif len(selected) > pool_size:
        removed = len(selected) - pool_size
        keep = sorted(
            list(selected),
            key=lambda q: (STRATA_ORDER.get(meta[q].stratum, 9), -meta[q].difficulty, q),
        )[:pool_size]
        selected = set(keep)
        fallbacks.append(f"pool_trimmed_to_size: -{removed}")

    return selected


def _rebalance_yesno(
    selected: set[str],
    meta: dict[str, QidMeta],
    all_qids: list[str],
    target_yesno: int,
    tol: int,
    fallbacks: list[str],
    hard100_set: set[str] | None = None,
) -> None:
    min_yesno = max(0, target_yesno - tol)
    max_yesno = target_yesno + tol

    non_selected = set(all_qids) - selected

    def current_yesno() -> int:
        return sum(1 for q in selected if meta[q].answer_type == "yesno")

    attempts = 0
    while attempts < 5000:
        attempts += 1
        y = current_yesno()
        if min_yesno <= y <= max_yesno:
            return
        if y < min_yesno:
            add = [q for q in non_selected if meta[q].answer_type == "yesno"]
            rem = [q for q in selected if meta[q].answer_type != "yesno"]
        else:
            add = [q for q in non_selected if meta[q].answer_type != "yesno"]
            rem = [q for q in selected if meta[q].answer_type == "yesno"]
        if hard100_set:
            rem = [q for q in rem if q not in hard100_set]
        add_sorted = sorted(
            add,
            key=lambda q: (
                STRATA_ORDER.get(meta[q].stratum, 9),
                -meta[q].difficulty,
                q,
            ),
        )
        rem_sorted = sorted(
            rem,
            key=lambda q: (
                STRATA_ORDER.get(meta[q].stratum, 9),
                meta[q].difficulty,
                q,
            ),
        )
        swapped = False
        for q_add in add_sorted:
            for q_rem in rem_sorted:
                if meta[q_add].stratum != meta[q_rem].stratum:
                    continue
                selected.remove(q_rem)
                selected.add(q_add)
                non_selected.remove(q_add)
                non_selected.add(q_rem)
                swapped = True
                break
            if swapped:
                break
        if not swapped:
            break

    final_yesno = current_yesno()
    if not (min_yesno <= final_yesno <= max_yesno):
        fallbacks.append(
            f"yesno_range_unmet: target={target_yesno}±{tol}, final={final_yesno}"
        )


def _rebalance_hard100(
    selected: set[str],
    meta: dict[str, QidMeta],
    all_qids: list[str],
    hard100_set: set[str],
    target: int,
    fallbacks: list[str],
) -> None:
    non_selected = set(all_qids) - selected

    def current() -> int:
        return sum(1 for q in selected if q in hard100_set)

    if current() >= target:
        return

    add_cands = sorted(
        [q for q in non_selected if q in hard100_set],
        key=lambda q: (STRATA_ORDER.get(meta[q].stratum, 9), -meta[q].difficulty, q),
    )
    rem_by_stratum: dict[str, list[str]] = {}
    for q in selected:
        if q in hard100_set:
            continue
        s = meta[q].stratum
        rem_by_stratum.setdefault(s, []).append(q)
    for s in rem_by_stratum:
        rem_by_stratum[s].sort(
            key=lambda q: (
                1 if meta[q].answer_type == "yesno" else 0,
                meta[q].difficulty,
                q,
            )
        )

    for q_add in add_cands:
        if current() >= target:
            break
        s = meta[q_add].stratum
        rem_list = rem_by_stratum.get(s, [])
        if not rem_list:
            continue
        q_rem = rem_list.pop(0)
        selected.remove(q_rem)
        selected.add(q_add)
        non_selected.remove(q_add)
        non_selected.add(q_rem)

    final = current()
    if final < target:
        fallbacks.append(f"hard100_target_unmet_after_swap: target={target}, final={final}")


def _rebalance_qtypes(
    selected: set[str],
    meta: dict[str, QidMeta],
    all_qids: list[str],
    qtype_ranges: dict[str, tuple[int, int]],
    fallbacks: list[str],
    hard100_set: set[str] | None = None,
) -> None:
    non_selected = set(all_qids) - selected

    def q_counts() -> Counter[str]:
        c = Counter()
        for q in selected:
            c[meta[q].qtype] += 1
        return c

    attempts = 0
    while attempts < 10000:
        attempts += 1
        counts = q_counts()
        under = [qt for qt, (lo, _) in qtype_ranges.items() if counts.get(qt, 0) < lo]
        over = [qt for qt, (_, hi) in qtype_ranges.items() if counts.get(qt, 0) > hi]
        if not under and not over:
            return
        if not under or not over:
            break

        qt_under = sorted(under, key=lambda q: counts.get(q, 0) - qtype_ranges[q][0])[0]
        qt_over = sorted(over, key=lambda q: counts.get(q, 0) - qtype_ranges[q][1], reverse=True)[0]

        add_cands = sorted(
            [q for q in non_selected if meta[q].qtype == qt_under],
            key=lambda q: (STRATA_ORDER.get(meta[q].stratum, 9), -meta[q].difficulty, q),
        )
        rem_cands = sorted(
            [
                q
                for q in selected
                if meta[q].qtype == qt_over and (not hard100_set or q not in hard100_set)
            ],
            key=lambda q: (STRATA_ORDER.get(meta[q].stratum, 9), meta[q].difficulty, q),
        )

        swapped = False
        for q_add in add_cands:
            for q_rem in rem_cands:
                if meta[q_add].stratum != meta[q_rem].stratum:
                    continue
                selected.remove(q_rem)
                selected.add(q_add)
                non_selected.remove(q_add)
                non_selected.add(q_rem)
                swapped = True
                break
            if swapped:
                break
        if not swapped:
            break

    counts = q_counts()
    unresolved = []
    for qt, (lo, hi) in qtype_ranges.items():
        n = counts.get(qt, 0)
        if n < lo or n > hi:
            unresolved.append(f"{qt}:{n} not in [{lo},{hi}]")
    if unresolved:
        fallbacks.append("qtype_range_unmet: " + ", ".join(unresolved))


def _build_meta(
    static_rows: dict[str, dict[str, Any]],
    adaptive_rows: dict[str, dict[str, Any]],
    bm25_rows: dict[str, dict[str, Any]],
) -> tuple[dict[str, QidMeta], dict[str, float]]:
    common = sorted(set(static_rows) & set(adaptive_rows) & set(bm25_rows))
    out: dict[str, QidMeta] = {}
    difficulties: list[float] = []

    for qid in common:
        rows = [static_rows[qid], adaptive_rows[qid], bm25_rows[qid]]

        f1_vals = []
        pair10_vals = []
        support2_vals = []
        em_vals = []
        question = str(rows[0].get("question", ""))
        refs = rows[0].get("answer_refs", [])

        for row in rows:
            am = row.get("answer_metrics_per_query", {}) if isinstance(row.get("answer_metrics_per_query"), dict) else {}
            f1_vals.append(_safe_float(am.get("f1", 0.0)))
            em_vals.append(_safe_float(am.get("em", 0.0)))
            prm = am.get("pair_recall_metrics", {}) if isinstance(am.get("pair_recall_metrics"), dict) else {}
            sdm = am.get("support_doc_metrics", {}) if isinstance(am.get("support_doc_metrics"), dict) else {}
            pair10_vals.append(_safe_float(prm.get("pair_recall@10", 0.0)))
            support2_vals.append(_safe_float(sdm.get("support_doc_recall@2", 0.0)))

        avg_f1 = sum(f1_vals) / 3.0
        avg_pair10 = sum(pair10_vals) / 3.0
        avg_support2 = sum(support2_vals) / 3.0
        all_em0 = int(all(x <= 0.0 for x in em_vals))
        difficulty = (
            0.45 * (1.0 - avg_f1)
            + 0.25 * (1.0 - avg_pair10)
            + 0.20 * (1.0 - avg_support2)
            + 0.10 * all_em0
        )
        difficulties.append(float(difficulty))

        out[qid] = QidMeta(
            qid=qid,
            question=question,
            qtype=_first_token(question),
            answer_type=_answer_type(refs),
            avg_f1=float(avg_f1),
            avg_pair10=float(avg_pair10),
            avg_support2=float(avg_support2),
            all_em0=int(all_em0),
            difficulty=float(difficulty),
            stratum="mid",
        )

    p35 = _pct(difficulties, 0.35)
    p75 = _pct(difficulties, 0.75)
    for m in out.values():
        if m.difficulty >= p75:
            m.stratum = "hard"
        elif m.difficulty >= p35:
            m.stratum = "mid"
        else:
            m.stratum = "easy" if m.all_em0 == 0 else "mid"

    return out, {"p35": p35, "p75": p75}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build representative HotpotQA pool from full 7405 runs.")
    parser.add_argument(
        "--full-static-per-query",
        default="results/full_static_51ac28f2d37b_p4/cfg_7c5052f7a8f9/hotpot_qa/per_query.jsonl",
    )
    parser.add_argument(
        "--full-adaptive-per-query",
        default="results/full_adaptive_51ac28f2d37b_p4/cfg_6c004a30aae6/hotpot_qa/per_query.jsonl",
    )
    parser.add_argument(
        "--full-bm25-per-query",
        default="results/baseline_full_bm25only_p4_sudo/cfg_dafbe591970e/hotpot_qa/per_query.jsonl",
    )
    parser.add_argument("--hard100-qids", default="qids/hotpot_hard100_lowerbound.json")
    parser.add_argument("--pool-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-qids", default="qids/hotpot_repr300_v1.json")
    parser.add_argument("--out-report", default="results/pool_repr300_v1_report.json")
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    static_path = Path(args.full_static_per_query)
    adaptive_path = Path(args.full_adaptive_per_query)
    bm25_path = Path(args.full_bm25_per_query)
    hard100_path = Path(args.hard100_qids)

    static_rows = _qid_map(static_path)
    adaptive_rows = _qid_map(adaptive_path)
    bm25_rows = _qid_map(bm25_path)
    meta, thresholds = _build_meta(static_rows, adaptive_rows, bm25_rows)
    all_qids = sorted(meta.keys())

    if hard100_path.exists():
        hard100 = json.loads(hard100_path.read_text(encoding="utf-8"))
        hard100 = [str(x) for x in hard100 if str(x).strip()]
    else:
        hard100 = []
    hard100_set = set(hard100)

    fallbacks: list[str] = []
    selected = _select_initial(
        meta=meta,
        hard100=hard100,
        pool_size=int(args.pool_size),
        rng=rng,
        fallbacks=fallbacks,
    )

    _rebalance_hard100(
        selected=selected,
        meta=meta,
        all_qids=all_qids,
        hard100_set=hard100_set,
        target=80,
        fallbacks=fallbacks,
    )

    # Anti-bias constraints.
    yesno_target = 19
    yesno_tol = 4
    _rebalance_yesno(
        selected=selected,
        meta=meta,
        all_qids=all_qids,
        target_yesno=yesno_target,
        tol=yesno_tol,
        fallbacks=fallbacks,
        hard100_set=hard100_set,
    )

    full_qtypes = Counter(meta[q].qtype for q in all_qids)
    q_targets = _qtype_targets(
        full_counts=full_qtypes,
        full_total=len(all_qids),
        pool_size=int(args.pool_size),
        qtypes=TOP_QTYPES,
    )
    q_ranges = _compute_ranges(q_targets, rel_tol=0.20)
    _rebalance_qtypes(
        selected=selected,
        meta=meta,
        all_qids=all_qids,
        qtype_ranges=q_ranges,
        fallbacks=fallbacks,
        hard100_set=hard100_set,
    )

    # Enforce hard100 inclusion as a hard constraint after anti-bias swaps.
    _rebalance_hard100(
        selected=selected,
        meta=meta,
        all_qids=all_qids,
        hard100_set=hard100_set,
        target=80,
        fallbacks=fallbacks,
    )

    if not _yesno_in_range(selected, meta, yesno_target, yesno_tol):
        yesno_count = sum(1 for q in selected if meta[q].answer_type == "yesno")
        fallbacks.append(
            f"yesno_range_unmet_final: target={yesno_target}±{yesno_tol}, final={yesno_count}"
        )
    qtype_ok_final, qtype_violations = _qtypes_in_range(selected, meta, q_ranges)
    if not qtype_ok_final:
        fallbacks.append("qtype_range_unmet_final: " + ", ".join(qtype_violations))

    selected_rows = sorted(
        [meta[q] for q in selected],
        key=lambda m: (STRATA_ORDER.get(m.stratum, 9), -m.difficulty, m.qid),
    )
    qids_out = [m.qid for m in selected_rows]

    selected_stats = _collect_stats(set(qids_out), meta)
    full_stats = _collect_stats(set(all_qids), meta)
    hard100_selected = [q for q in qids_out if q in hard100_set]
    yesno_selected = int(selected_stats.get("answer_type", {}).get("yesno", 0))
    qtype_ok, qtype_violations = _qtypes_in_range(set(qids_out), meta, q_ranges)
    constraints = {
        "pool_size_ok": bool(len(qids_out) == int(args.pool_size)),
        "unique_qids": bool(len(qids_out) == len(set(qids_out))),
        "hard_quota_ok": bool(int(selected_stats.get("strata", {}).get("hard", 0)) == 120),
        "mid_quota_ok": bool(int(selected_stats.get("strata", {}).get("mid", 0)) == 105),
        "easy_quota_ok": bool(int(selected_stats.get("strata", {}).get("easy", 0)) == 75),
        "hard100_selected_ge_80": bool(len(hard100_selected) >= 80),
        "yesno_range_ok": bool(max(0, yesno_target - yesno_tol) <= yesno_selected <= yesno_target + yesno_tol),
        "qtype_ranges_ok": bool(qtype_ok),
        "qtype_violations": qtype_violations,
    }

    report = {
        "task": "build_representative_pool",
        "seed": int(args.seed),
        "pool_size": int(args.pool_size),
        "difficulty_formula": "0.45*(1-avg_f1)+0.25*(1-avg_pair10)+0.20*(1-avg_support2)+0.10*all_em0",
        "source_paths": {
            "full_static_per_query": str(static_path),
            "full_adaptive_per_query": str(adaptive_path),
            "full_bm25_per_query": str(bm25_path),
            "hard100_qids": str(hard100_path),
        },
        "thresholds": thresholds,
        "targets": {
            "strata": {"hard": 120, "mid": 105, "easy": 75},
            "hard100_forced": 80,
            "yesno_target": int(yesno_target),
            "yesno_range": [max(0, yesno_target - yesno_tol), yesno_target + yesno_tol],
            "qtype_targets_top": q_targets,
            "qtype_ranges_top": {k: [v[0], v[1]] for k, v in q_ranges.items()},
        },
        "counts_full": full_stats,
        "counts_selected": selected_stats,
        "constraints": constraints,
        "hard100": {
            "available": int(len([q for q in hard100 if q in meta])),
            "selected": int(len(hard100_selected)),
        },
        "fallbacks": fallbacks,
        "selected_qids_path": str(args.out_qids),
        "selected_qids_count": int(len(qids_out)),
        "selected_preview_top20": [
            {
                "qid": m.qid,
                "stratum": m.stratum,
                "difficulty": m.difficulty,
                "avg_f1": m.avg_f1,
                "avg_pair10": m.avg_pair10,
                "avg_support2": m.avg_support2,
                "qtype": m.qtype,
                "answer_type": m.answer_type,
            }
            for m in selected_rows[:20]
        ],
        "selected_metadata": [
            {
                "qid": m.qid,
                "stratum": m.stratum,
                "difficulty": m.difficulty,
                "avg_f1": m.avg_f1,
                "avg_pair10": m.avg_pair10,
                "avg_support2": m.avg_support2,
                "all_em0": m.all_em0,
                "qtype": m.qtype,
                "answer_type": m.answer_type,
            }
            for m in selected_rows
        ],
    }

    save_json(args.out_qids, qids_out)
    save_json(args.out_report, report)

    print(f"Done. qids={args.out_qids} (n={len(qids_out)})")
    print(f"Report: {args.out_report}")


if __name__ == "__main__":
    main()
