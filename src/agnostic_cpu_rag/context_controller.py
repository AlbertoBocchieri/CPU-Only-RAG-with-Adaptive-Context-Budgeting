from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any

from .records import CoverageGoal, RetrievedCandidate
from .utils import clamp, head_words, jaccard_tokens, mad, minmax_normalize


@dataclass(slots=True)
class ContextSelectionResult:
    contexts: list[str]
    selected_candidates: list[RetrievedCandidate]
    trace: dict[str, Any]


class ContextController:
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.stop_mode = str(cfg.get("stop_mode", "coverage_locked_patience_v2"))
        self.seed_min_items = int(cfg.get("seed_min_items", 3))
        self.snippet_words = int(cfg.get("snippet_words", 120))
        self.min_snippet_words = int(cfg.get("min_snippet_words", 40))
        self.utility_weights = dict(
            cfg.get(
                "utility_weights",
                {
                    "relevance": 0.55,
                    "question_overlap": 0.20,
                    "novelty": 0.15,
                    "new_doc_bonus": 0.10,
                },
            )
        )
        self.patience = max(1, int(cfg.get("patience", 2)))
        self.single_evidence_extra_probe_candidates = max(
            0, int(cfg.get("single_evidence_extra_probe_candidates", 0))
        )
        self.multi_document_spare_probe_candidates = max(
            0, int(cfg.get("multi_document_spare_probe_candidates", 1))
        )
        self.multi_document_exact_probe_candidates = max(
            0, int(cfg.get("multi_document_exact_probe_candidates", 2))
        )
        self.marginal_snippet_enabled = bool(cfg.get("marginal_snippet_enabled", False))
        self.marginal_snippet_ratio = float(cfg.get("marginal_snippet_ratio", 0.60))
        self.marginal_snippet_min_words = int(cfg.get("marginal_snippet_min_words", 40))
        self.marginal_snippet_max_words = int(cfg.get("marginal_snippet_max_words", 72))

    def _extra_probe_plan(
        self,
        coverage_goal: CoverageGoal,
        seed_distinct_doc_count: int,
        required_distinct_docs: int,
    ) -> tuple[int, str]:
        if coverage_goal == CoverageGoal.RETRIEVAL_ONLY:
            return 0, "retrieval_only"
        if coverage_goal == CoverageGoal.SINGLE_EVIDENCE:
            return int(self.single_evidence_extra_probe_candidates), "single_evidence"
        if seed_distinct_doc_count >= max(0, int(required_distinct_docs)) + 1:
            return int(self.multi_document_spare_probe_candidates), "seed_has_spare_doc"
        return int(self.multi_document_exact_probe_candidates), "seed_exact_coverage"

    @staticmethod
    def _legacy_extra_probe_candidates(coverage_goal: CoverageGoal) -> int:
        if coverage_goal == CoverageGoal.MULTI_DOCUMENT_EVIDENCE:
            return 2
        if coverage_goal == CoverageGoal.SINGLE_EVIDENCE:
            return 1
        return 0

    @staticmethod
    def _pack_text(
        text: str,
        remaining_tokens: int,
        snippet_words: int,
        min_snippet_words: int,
        *,
        force_snippet: bool = False,
    ) -> tuple[str, int, str]:
        words = str(text).split()
        full_tokens = len(words)
        if not force_snippet and full_tokens <= max(0, int(remaining_tokens)):
            return str(text).strip(), int(full_tokens), "full"
        take = min(int(snippet_words), max(0, int(remaining_tokens)))
        if take < max(1, int(min_snippet_words)):
            return "", 0, "skipped"
        if take >= full_tokens:
            return str(text).strip(), int(full_tokens), "full"
        packed = " ".join(words[:take]).strip()
        return packed, int(take), "snippet"

    def _marginal_snippet_words(self, seed_median_tokens: int) -> int:
        scaled = round(float(seed_median_tokens) * float(self.marginal_snippet_ratio))
        return int(
            clamp(
                scaled,
                int(self.marginal_snippet_min_words),
                int(self.marginal_snippet_max_words),
            )
        )

    def _candidate_features(
        self,
        *,
        query: str,
        candidate: RetrievedCandidate,
        normalized_scores: dict[str, float],
        selected_candidates: list[RetrievedCandidate],
        seen_doc_ids: set[str],
    ) -> dict[str, float]:
        relevance = float(normalized_scores.get(candidate.item_id, 0.0))
        query_overlap = jaccard_tokens(query, head_words(candidate.text, 80))
        if not selected_candidates:
            novelty = 1.0
        else:
            novelty = 1.0 - max(
                jaccard_tokens(head_words(candidate.text, 80), head_words(prev.text, 80))
                for prev in selected_candidates
            )
        novelty = clamp(novelty, 0.0, 1.0)
        new_doc_bonus = 1.0 if candidate.doc_id not in seen_doc_ids else 0.0
        weights = self.utility_weights
        utility = (
            float(weights.get("relevance", 0.55)) * relevance
            + float(weights.get("question_overlap", 0.20)) * query_overlap
            + float(weights.get("novelty", 0.15)) * novelty
            + float(weights.get("new_doc_bonus", 0.10)) * new_doc_bonus
        )
        return {
            "relevance": relevance,
            "question_overlap": float(query_overlap),
            "novelty": float(novelty),
            "new_doc_bonus": float(new_doc_bonus),
            "utility": float(utility),
        }

    def select(
        self,
        *,
        query: str,
        candidates: list[RetrievedCandidate],
        coverage_goal: CoverageGoal,
        required_distinct_docs: int,
        budget_cap_tokens: int,
    ) -> ContextSelectionResult:
        if not self.enabled or not candidates:
            contexts = [c.text for c in candidates]
            return ContextSelectionResult(
                contexts=contexts,
                selected_candidates=list(candidates),
                trace={
                    "enabled": False,
                    "stop_mode": self.stop_mode,
                    "coverage_goal": str(coverage_goal.value),
                    "context_tokens_used": int(sum(len(c.text.split()) for c in candidates)),
                    "budget_cap_tokens": int(budget_cap_tokens),
                    "selected_count": len(candidates),
                    "candidate_trace": [],
                },
            )

        score_map = {c.item_id: float(c.score) for c in candidates}
        norm_scores = minmax_normalize(score_map)
        selected_candidates: list[RetrievedCandidate] = []
        selected_contexts: list[str] = []
        candidate_trace: list[dict[str, Any]] = []
        seen_doc_ids: set[str] = set()
        used_tokens = 0
        required_distinct_docs = max(0, int(required_distinct_docs))
        seed_target = max(self.seed_min_items, required_distinct_docs + 1)

        for rank, candidate in enumerate(candidates, start=1):
            features = self._candidate_features(
                query=query,
                candidate=candidate,
                normalized_scores=norm_scores,
                selected_candidates=selected_candidates,
                seen_doc_ids=seen_doc_ids,
            )
            remaining = max(0, int(budget_cap_tokens) - int(used_tokens))
            packed_text, packed_tokens, packing_mode = self._pack_text(
                candidate.text,
                remaining_tokens=remaining,
                snippet_words=self.snippet_words,
                min_snippet_words=self.min_snippet_words,
            )
            if packed_tokens <= 0:
                break
            selected_candidates.append(candidate)
            selected_contexts.append(packed_text)
            seen_doc_ids.add(candidate.doc_id)
            used_tokens += packed_tokens
            candidate_trace.append(
                {
                    "rank": int(rank),
                    "item_id": candidate.item_id,
                    "doc_id": candidate.doc_id,
                    "packing_mode": packing_mode,
                    "tokens_selected": int(packed_tokens),
                    "full_tokens": int(len(candidate.text.split())),
                    "selected": True,
                    "stop_rule": "seed",
                    "stop_decision": False,
                    **features,
                }
            )
            if len(selected_candidates) >= seed_target and len(seen_doc_ids) >= max(1, required_distinct_docs):
                break

        seed_utilities = [float(row["utility"]) for row in candidate_trace] if candidate_trace else [0.0]
        seed_novelties = [float(row["novelty"]) for row in candidate_trace] if candidate_trace else [1.0]
        seed_overlaps = [float(row["question_overlap"]) for row in candidate_trace] if candidate_trace else [0.0]
        seed_relevances = [float(row["relevance"]) for row in candidate_trace] if candidate_trace else [0.0]
        seed_token_counts = [int(row["tokens_selected"]) for row in candidate_trace] if candidate_trace else [0]
        theta_q = float(median(seed_utilities) - mad(seed_utilities))
        seed_median_relevance = float(median(seed_relevances))
        seed_median_novelty = float(median(seed_novelties))
        seed_median_overlap = float(median(seed_overlaps))
        seed_median_tokens = int(median(seed_token_counts)) if seed_token_counts else 0
        seed_distinct_doc_count = int(len({row["doc_id"] for row in candidate_trace[: len(seed_token_counts)]}))
        if self.stop_mode == "coverage_locked_patience_v3":
            extra_probe_candidates, extra_probe_reason = self._extra_probe_plan(
                coverage_goal,
                seed_distinct_doc_count,
                required_distinct_docs,
            )
        else:
            extra_probe_candidates = self._legacy_extra_probe_candidates(coverage_goal)
            extra_probe_reason = "legacy_fixed"
        unlock_min_items = int(len(selected_candidates) + extra_probe_candidates)
        if self.stop_mode in {"coverage_locked_patience_v2", "coverage_locked_patience_v3"}:
            dynamic_floor_tokens = int(sum(seed_token_counts) + (extra_probe_candidates * seed_median_tokens))
        else:
            dynamic_floor_tokens = int(
                max(
                    sum(seed_token_counts),
                    max(1, required_distinct_docs) * seed_median_tokens if seed_token_counts else 0,
                )
            )
        stop_counter = 0
        marginal_snippets_applied = 0

        for rank, candidate in enumerate(candidates[len(selected_candidates) :], start=len(selected_candidates) + 1):
            features = self._candidate_features(
                query=query,
                candidate=candidate,
                normalized_scores=norm_scores,
                selected_candidates=selected_candidates,
                seen_doc_ids=seen_doc_ids,
            )
            coverage_unlocked = bool(
                len(seen_doc_ids) >= required_distinct_docs
                and used_tokens >= dynamic_floor_tokens
                and len(selected_candidates) >= unlock_min_items
            )

            stop_rule = self.stop_mode
            stop_decision = False
            stop_before = int(stop_counter)
            low_utility = False
            reset = False
            if coverage_unlocked:
                if self.stop_mode == "utility_only":
                    stop_decision = bool(float(features["utility"]) < theta_q)
                elif self.stop_mode == "triple_condition":
                    stop_decision = bool(
                        float(features["utility"]) < theta_q
                        and float(features["novelty"]) < seed_median_novelty
                        and (
                            candidate.doc_id in seen_doc_ids
                            or float(features["question_overlap"]) < seed_median_overlap
                        )
                    )
                elif self.stop_mode == "coverage_locked_patience":
                    low_utility = bool(float(features["utility"]) < theta_q)
                    # Reset uses the raw relevance component only, not the composed utility.
                    # This keeps the patience rule from being relaxed by novelty/overlap terms.
                    reset = bool(
                        float(features["utility"]) >= theta_q
                        or (
                            candidate.doc_id not in seen_doc_ids
                            and float(features["relevance"]) >= seed_median_relevance
                        )
                    )
                    if reset:
                        stop_counter = 0
                    elif low_utility:
                        stop_counter += 1
                    stop_decision = bool(stop_counter >= self.patience)
                else:
                    low_utility = bool(float(features["utility"]) < theta_q)
                    reset = bool(
                        float(features["utility"]) >= theta_q
                        or (
                            candidate.doc_id not in seen_doc_ids
                            and float(features["relevance"]) >= seed_median_relevance
                        )
                    )
                    if reset:
                        stop_counter = 0
                    elif low_utility:
                        stop_counter += 1
                    stop_decision = bool(stop_counter >= self.patience)
            stop_after = int(stop_counter)

            candidate_full_tokens = int(len(candidate.text.split()))
            marginal_snippet_candidate = bool(
                self.marginal_snippet_enabled
                and coverage_goal == CoverageGoal.MULTI_DOCUMENT_EVIDENCE
                and coverage_unlocked
                and low_utility
                and not reset
                and not stop_decision
            )
            remaining = max(0, int(budget_cap_tokens) - int(used_tokens))
            marginal_snippet_words = self._marginal_snippet_words(seed_median_tokens)
            force_marginal_snippet = bool(marginal_snippet_candidate)
            packing_policy_reason = "seed_full"
            if coverage_unlocked:
                if force_marginal_snippet:
                    packing_policy_reason = "post_unlock_marginal_snippet"
                else:
                    packing_policy_reason = "post_unlock_strong_full"

            packed_text, packed_tokens, packing_mode = self._pack_text(
                candidate.text,
                remaining_tokens=remaining,
                snippet_words=(marginal_snippet_words if force_marginal_snippet else self.snippet_words),
                min_snippet_words=(
                    self.marginal_snippet_min_words if force_marginal_snippet else self.min_snippet_words
                ),
                force_snippet=force_marginal_snippet,
            )
            if packed_tokens <= 0:
                candidate_trace.append(
                    {
                        "rank": int(rank),
                        "item_id": candidate.item_id,
                        "doc_id": candidate.doc_id,
                        "packing_mode": "skipped",
                        "packing_policy_reason": "budget_skipped",
                        "marginal_snippet_applied": bool(force_marginal_snippet),
                        "marginal_snippet_words": int(marginal_snippet_words if force_marginal_snippet else 0),
                        "tokens_selected": 0,
                        "full_tokens": int(candidate_full_tokens),
                        "selected": False,
                        "coverage_unlocked": bool(coverage_unlocked),
                        "stop_rule": stop_rule,
                        "stop_counter_before": stop_before,
                        "stop_counter_after": stop_after,
                        "stop_decision": bool(stop_decision),
                        **features,
                    }
                )
                break
            if force_marginal_snippet and packing_mode == "snippet":
                marginal_snippets_applied += 1
            elif packing_mode == "snippet":
                packing_policy_reason = "budget_forced_snippet"
            candidate_trace.append(
                {
                    "rank": int(rank),
                    "item_id": candidate.item_id,
                    "doc_id": candidate.doc_id,
                    "packing_mode": packing_mode,
                    "packing_policy_reason": packing_policy_reason,
                    "marginal_snippet_applied": bool(force_marginal_snippet and packing_mode == "snippet"),
                    "marginal_snippet_words": int(marginal_snippet_words if force_marginal_snippet else 0),
                    "tokens_selected": int(packed_tokens),
                    "full_tokens": int(candidate_full_tokens),
                    "selected": not stop_decision,
                    "coverage_unlocked": bool(coverage_unlocked),
                    "stop_rule": stop_rule,
                    "stop_counter_before": stop_before,
                    "stop_counter_after": stop_after,
                    "stop_decision": bool(stop_decision),
                    **features,
                }
            )
            if stop_decision:
                break
            selected_candidates.append(candidate)
            selected_contexts.append(packed_text)
            seen_doc_ids.add(candidate.doc_id)
            used_tokens += packed_tokens

        trace = {
            "enabled": True,
            "stop_mode": self.stop_mode,
            "coverage_goal": str(coverage_goal.value),
            "required_distinct_docs": int(required_distinct_docs),
            "coverage_unlocked": bool(
                len(seen_doc_ids) >= required_distinct_docs
                and used_tokens >= dynamic_floor_tokens
                and len(selected_candidates) >= unlock_min_items
            ),
            "seed_item_count": int(len(seed_token_counts)),
            "seed_distinct_doc_count": int(seed_distinct_doc_count),
            "seed_token_total": int(sum(seed_token_counts)),
            "seed_median_tokens": int(seed_median_tokens),
            "extra_probe_candidates": int(extra_probe_candidates),
            "extra_probe_reason": str(extra_probe_reason),
            "unlock_min_items": int(unlock_min_items),
            "dynamic_floor_tokens": int(dynamic_floor_tokens),
            "theta_q": float(theta_q),
            "context_tokens_used": int(used_tokens),
            "budget_cap_tokens": int(budget_cap_tokens),
            "single_evidence_extra_probe_candidates": int(self.single_evidence_extra_probe_candidates),
            "multi_document_spare_probe_candidates": int(self.multi_document_spare_probe_candidates),
            "multi_document_exact_probe_candidates": int(self.multi_document_exact_probe_candidates),
            "marginal_snippet_enabled": bool(self.marginal_snippet_enabled),
            "marginal_snippet_ratio": float(self.marginal_snippet_ratio),
            "marginal_snippet_min_words": int(self.marginal_snippet_min_words),
            "marginal_snippet_max_words": int(self.marginal_snippet_max_words),
            "marginal_snippets_applied": int(marginal_snippets_applied),
            "selected_count": len(selected_candidates),
            "selected_doc_count": len(seen_doc_ids),
            "context_item_ids_used": [c.item_id for c in selected_candidates],
            "context_doc_ids_used": list(dict.fromkeys(c.doc_id for c in selected_candidates)),
            "candidate_trace": candidate_trace,
            "seed_stats": {
                "seed_utilities": seed_utilities,
                "seed_median_relevance": float(seed_median_relevance),
                "seed_median_novelty": float(seed_median_novelty),
                "seed_median_question_overlap": float(seed_median_overlap),
            },
            "utility_weights": dict(self.utility_weights),
        }
        return ContextSelectionResult(
            contexts=selected_contexts,
            selected_candidates=selected_candidates,
            trace=trace,
        )
