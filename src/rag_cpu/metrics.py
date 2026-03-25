from __future__ import annotations

import math
import re
import string
from collections import defaultdict
from statistics import mean
from typing import Iterable


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_ARTICLES_RE = re.compile(r"\b(a|an|the|il|lo|la|i|gli|le|un|uno|una|dei|degli|delle|del|della)\b")
_WS_RE = re.compile(r"\s+")


def normalize_answer(s: str) -> str:
    s = s.lower().translate(_PUNCT_TABLE)
    s = _ARTICLES_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = defaultdict(int)
    for t in gt_tokens:
        common[t] += 1

    overlap = 0
    for t in pred_tokens:
        if common[t] > 0:
            common[t] -= 1
            overlap += 1

    if overlap == 0:
        return 0.0
    p = overlap / len(pred_tokens)
    r = overlap / len(gt_tokens)
    return 2 * p * r / (p + r)


def qa_scores(prediction: str, references: Iterable[str]) -> dict[str, float]:
    refs = list(references)
    if not refs:
        return {"em": 0.0, "f1": 0.0}
    em = max(exact_match_score(prediction, r) for r in refs)
    f1 = max(f1_score(prediction, r) for r in refs)
    return {"em": em, "f1": f1}


def evaluate_retrieval(
    runs: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    ks: list[int],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    per_query: dict[str, dict[str, float]] = {}
    metrics: dict[str, list[float]] = defaultdict(list)

    max_k = max(ks)
    for qid, rel in qrels.items():
        rel_docs = {doc_id: int(score) for doc_id, score in rel.items() if int(score) > 0}
        if not rel_docs:
            continue
        ranked = runs.get(qid, [])[:max_k]

        qvals: dict[str, float] = {}

        for k in ks:
            top = ranked[:k]
            hits = [doc for doc in top if doc in rel_docs]
            recall = len(hits) / max(1, len(rel_docs))
            precision = len(hits) / max(1, len(top))

            dcg = 0.0
            for i, doc_id in enumerate(top):
                rel_grade = rel_docs.get(doc_id, 0)
                if rel_grade > 0:
                    dcg += (2**rel_grade - 1) / math.log2(i + 2)
            ideal_grades = sorted(rel_docs.values(), reverse=True)[:k]
            idcg = 0.0
            for i, rel_grade in enumerate(ideal_grades):
                idcg += (2**rel_grade - 1) / math.log2(i + 2)
            ndcg = dcg / idcg if idcg > 0 else 0.0

            rr = 0.0
            for i, doc_id in enumerate(top):
                if doc_id in rel_docs:
                    rr = 1.0 / (i + 1)
                    break

            ap_hits = 0
            ap_acc = 0.0
            for i, doc_id in enumerate(top, start=1):
                if doc_id in rel_docs:
                    ap_hits += 1
                    ap_acc += ap_hits / i
            ap_denom = max(1, min(len(rel_docs), k))
            ap = ap_acc / ap_denom
            hitrate = 1.0 if hits else 0.0

            qvals[f"Recall@{k}"] = recall
            qvals[f"Precision@{k}"] = precision
            qvals[f"nDCG@{k}"] = ndcg
            qvals[f"MRR@{k}"] = rr
            qvals[f"MAP@{k}"] = ap
            qvals[f"HitRate@{k}"] = hitrate

            metrics[f"Recall@{k}"].append(recall)
            metrics[f"Precision@{k}"].append(precision)
            metrics[f"nDCG@{k}"].append(ndcg)
            metrics[f"MRR@{k}"].append(rr)
            metrics[f"MAP@{k}"].append(ap)
            metrics[f"HitRate@{k}"].append(hitrate)

        per_query[qid] = qvals

    agg = {k: mean(v) if v else 0.0 for k, v in metrics.items()}
    return agg, per_query


def summarize_list(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0}
    s = sorted(values)
    n = len(s)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(round((n - 1) * p))))
        return float(s[idx])

    return {"mean": float(mean(values)), "p50": pct(0.5), "p90": pct(0.9), "p95": pct(0.95)}
