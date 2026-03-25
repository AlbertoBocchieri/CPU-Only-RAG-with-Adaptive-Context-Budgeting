from __future__ import annotations

from dataclasses import dataclass

from rag_cpu.metrics import qa_scores

from ..records import CoverageGoal, GoldReference
from .base import TaskAdapter


@dataclass(slots=True)
class RetrievalOnlyTaskAdapter(TaskAdapter):
    name: str = "retrieval_only"
    family: str = "retrieval_only"
    coverage_goal: CoverageGoal = CoverageGoal.RETRIEVAL_ONLY
    supports_generation: bool = False

    def required_distinct_docs(self) -> int:
        return 0

    def evaluate_answer(self, prediction: str, gold: GoldReference) -> dict[str, float | None]:
        return {"em": None, "f1": None}

    def evaluate_context(self, selected_doc_ids: list[str], gold: GoldReference) -> dict[str, float | int | None]:
        selected = list(dict.fromkeys(selected_doc_ids))
        relevant = [doc_id for doc_id in selected if doc_id in gold.relevant_doc_ids]
        recall = len(set(relevant)) / max(1, len(gold.relevant_doc_ids)) if gold.relevant_doc_ids else None
        return {
            "selected_distinct_docs": len(selected),
            "relevant_doc_recall": recall,
            "coverage_goal_met": 1.0,
            "pair_in_context": None,
        }


@dataclass(slots=True)
class OpenQATaskAdapter(TaskAdapter):
    name: str = "open_qa"
    family: str = "open_qa"
    coverage_goal: CoverageGoal = CoverageGoal.SINGLE_EVIDENCE
    supports_generation: bool = True

    def required_distinct_docs(self) -> int:
        return 1

    def evaluate_answer(self, prediction: str, gold: GoldReference) -> dict[str, float | None]:
        scores = qa_scores(prediction, gold.answers)
        return {"em": float(scores["em"]), "f1": float(scores["f1"])}

    def evaluate_context(self, selected_doc_ids: list[str], gold: GoldReference) -> dict[str, float | int | None]:
        selected = list(dict.fromkeys(selected_doc_ids))
        relevant = [doc_id for doc_id in selected if doc_id in gold.relevant_doc_ids]
        recall = len(set(relevant)) / max(1, len(gold.relevant_doc_ids)) if gold.relevant_doc_ids else None
        coverage_goal_met = 1.0 if len(selected) >= 1 else 0.0
        return {
            "selected_distinct_docs": len(selected),
            "relevant_doc_recall": recall,
            "coverage_goal_met": coverage_goal_met,
            "pair_in_context": None,
        }


@dataclass(slots=True)
class MultiHopQATaskAdapter(TaskAdapter):
    name: str = "multi_hop_qa"
    family: str = "multi_hop_qa"
    coverage_goal: CoverageGoal = CoverageGoal.MULTI_DOCUMENT_EVIDENCE
    supports_generation: bool = True

    def required_distinct_docs(self) -> int:
        return 2

    def evaluate_answer(self, prediction: str, gold: GoldReference) -> dict[str, float | None]:
        scores = qa_scores(prediction, gold.answers)
        return {"em": float(scores["em"]), "f1": float(scores["f1"])}

    def evaluate_context(self, selected_doc_ids: list[str], gold: GoldReference) -> dict[str, float | int | None]:
        selected = list(dict.fromkeys(selected_doc_ids))
        relevant = [doc_id for doc_id in selected if doc_id in gold.relevant_doc_ids]
        recall = len(set(relevant)) / max(1, len(gold.relevant_doc_ids)) if gold.relevant_doc_ids else None
        pair_target = 2 if len(gold.relevant_doc_ids) >= 2 else min(1, len(gold.relevant_doc_ids))
        pair_hits = len(set(relevant))
        pair_in_context = None
        if pair_target > 0:
            pair_in_context = 1.0 if pair_hits >= pair_target else 0.0
        coverage_goal_met = 1.0 if len(selected) >= 2 else 0.0
        return {
            "selected_distinct_docs": len(selected),
            "relevant_doc_recall": recall,
            "coverage_goal_met": coverage_goal_met,
            "pair_in_context": pair_in_context,
        }


def make_task_adapter(task_family: str) -> TaskAdapter:
    key = str(task_family).strip().lower()
    if key == "multi_hop_qa":
        return MultiHopQATaskAdapter()
    if key == "open_qa":
        return OpenQATaskAdapter()
    if key == "retrieval_only":
        return RetrievalOnlyTaskAdapter()
    raise ValueError(f"Unsupported task family: {task_family}")
