from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_cpu.data import (
    load_beir_dataset,
    load_hotpotqa_distractor,
    load_natural_questions_validation,
    load_squad_open_validation,
    load_two_wiki_multihop_validation,
)

from ..records import DatasetBundle, DocumentRecord, GoldReference, QueryRecord
from .base import DatasetAdapter


def _convert_docs(docs: dict[str, Any]) -> dict[str, DocumentRecord]:
    return {
        doc_id: DocumentRecord(
            doc_id=str(doc.doc_id),
            title=str(getattr(doc, "title", "") or ""),
            text=str(doc.text),
            metadata=dict(getattr(doc, "metadata", {}) or {}),
        )
        for doc_id, doc in docs.items()
    }


def _convert_qa_examples(dataset_name: str, task_family_hint: str, examples: list[Any], docs: dict[str, Any]) -> DatasetBundle:
    queries: list[QueryRecord] = []
    gold: dict[str, GoldReference] = {}
    for ex in examples:
        queries.append(QueryRecord(qid=str(ex.qid), query=str(ex.question), metadata=dict(ex.metadata or {})))
        gold[str(ex.qid)] = GoldReference(
            qid=str(ex.qid),
            answers=[str(x) for x in list(ex.answers)],
            relevant_doc_ids={str(x) for x in set(ex.gold_doc_ids)},
            metadata=dict(ex.metadata or {}),
        )
    return DatasetBundle(
        dataset_name=dataset_name,
        task_family_hint=task_family_hint,
        queries=queries,
        documents=_convert_docs(docs),
        gold_references=gold,
        metadata={"num_docs": len(docs), "num_queries": len(queries)},
    )


@dataclass(slots=True)
class HotpotDatasetAdapter(DatasetAdapter):
    name: str = "hotpot_qa"
    default_split: str = "validation"
    task_family_hint: str = "multi_hop_qa"

    def load(
        self,
        *,
        max_queries: int,
        seed: int,
        include_qids: list[str] | None = None,
        split: str | None = None,
        **kwargs: Any,
    ) -> DatasetBundle:
        examples, docs = load_hotpotqa_distractor(
            split=str(split or self.default_split),
            max_queries=int(max_queries),
            seed=int(seed),
            include_qids=include_qids,
        )
        return _convert_qa_examples(self.name, self.task_family_hint, examples, docs)


@dataclass(slots=True)
class TwoWikiDatasetAdapter(DatasetAdapter):
    name: str = "two_wiki_multihop"
    default_split: str = "validation"
    task_family_hint: str = "multi_hop_qa"

    def load(
        self,
        *,
        max_queries: int,
        seed: int,
        include_qids: list[str] | None = None,
        split: str | None = None,
        **kwargs: Any,
    ) -> DatasetBundle:
        del split
        examples, docs = load_two_wiki_multihop_validation(
            max_queries=int(max_queries),
            seed=int(seed),
            include_qids=include_qids,
        )
        return _convert_qa_examples(self.name, self.task_family_hint, examples, docs)


@dataclass(slots=True)
class NaturalQuestionsDatasetAdapter(DatasetAdapter):
    name: str = "natural_questions"
    default_split: str = "validation"
    task_family_hint: str = "open_qa"

    def load(
        self,
        *,
        max_queries: int,
        seed: int,
        include_qids: list[str] | None = None,
        split: str | None = None,
        **kwargs: Any,
    ) -> DatasetBundle:
        del split
        examples, docs = load_natural_questions_validation(
            max_queries=int(max_queries),
            seed=int(seed),
            include_qids=include_qids,
            streaming=bool(kwargs.get("streaming", False)),
        )
        return _convert_qa_examples(self.name, self.task_family_hint, examples, docs)


@dataclass(slots=True)
class SquadOpenDatasetAdapter(DatasetAdapter):
    name: str = "squad_open"
    default_split: str = "validation"
    task_family_hint: str = "open_qa"

    def load(
        self,
        *,
        max_queries: int,
        seed: int,
        include_qids: list[str] | None = None,
        split: str | None = None,
        **kwargs: Any,
    ) -> DatasetBundle:
        del split, kwargs
        examples, docs = load_squad_open_validation(
            max_queries=int(max_queries),
            seed=int(seed),
            include_qids=include_qids,
        )
        return _convert_qa_examples(self.name, self.task_family_hint, examples, docs)


@dataclass(slots=True)
class BEIRDatasetAdapter(DatasetAdapter):
    name: str = "beir"
    default_split: str = "test"
    task_family_hint: str = "retrieval_only"
    dataset_name: str = "scifact"
    data_root: str = "data/beir"

    def load(
        self,
        *,
        max_queries: int,
        seed: int,
        include_qids: list[str] | None = None,
        split: str | None = None,
        **kwargs: Any,
    ) -> DatasetBundle:
        del split
        docs, queries, qrels = load_beir_dataset(
            dataset_name=str(kwargs.get("beir_name", self.dataset_name)),
            max_queries=int(max_queries),
            seed=int(seed),
            data_root=str(kwargs.get("data_root", self.data_root)),
        )
        query_records: list[QueryRecord] = []
        gold_references: dict[str, GoldReference] = {}
        keep_qids = include_qids if include_qids else list(queries.keys())
        for qid in keep_qids:
            if qid not in queries or qid not in qrels:
                continue
            query_records.append(QueryRecord(qid=str(qid), query=str(queries[qid]), metadata={"dataset": self.name}))
            gold_references[str(qid)] = GoldReference(
                qid=str(qid),
                answers=[],
                relevant_doc_ids={str(doc_id) for doc_id, grade in qrels[qid].items() if int(grade) > 0},
                metadata={"dataset": self.name, "beir_dataset": str(kwargs.get("beir_name", self.dataset_name))},
            )
        return DatasetBundle(
            dataset_name=self.name,
            task_family_hint=self.task_family_hint,
            queries=query_records,
            documents=_convert_docs(docs),
            gold_references=gold_references,
            metadata={
                "num_docs": len(docs),
                "num_queries": len(query_records),
                "beir_dataset": str(kwargs.get("beir_name", self.dataset_name)),
            },
        )


def make_dataset_adapter(dataset_name: str, **kwargs: Any) -> DatasetAdapter:
    key = str(dataset_name).strip().lower()
    if key == "hotpot_qa":
        return HotpotDatasetAdapter()
    if key == "two_wiki_multihop":
        return TwoWikiDatasetAdapter()
    if key == "natural_questions":
        return NaturalQuestionsDatasetAdapter()
    if key == "squad_open":
        return SquadOpenDatasetAdapter()
    if key == "beir":
        return BEIRDatasetAdapter(
            dataset_name=str(kwargs.get("beir_name", "scifact")),
            data_root=str(kwargs.get("data_root", "data/beir")),
        )
    raise ValueError(f"Unsupported dataset adapter: {dataset_name}")
