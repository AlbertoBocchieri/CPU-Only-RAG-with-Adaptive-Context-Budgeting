from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class CoverageGoal(str, Enum):
    RETRIEVAL_ONLY = "retrieval_only"
    SINGLE_EVIDENCE = "single_evidence"
    MULTI_DOCUMENT_EVIDENCE = "multi_document_evidence"


@dataclass(slots=True)
class DocumentRecord:
    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueryRecord:
    qid: str
    query: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GoldReference:
    qid: str
    answers: list[str] = field(default_factory=list)
    relevant_doc_ids: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedCandidate:
    item_id: str
    doc_id: str
    text: str
    score: float
    source: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetBundle:
    dataset_name: str
    task_family_hint: str
    queries: list[QueryRecord]
    documents: dict[str, DocumentRecord]
    gold_references: dict[str, GoldReference]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunManifestV2:
    run_id: str
    dataset: str
    task_family: str
    pool_role: str
    split: str
    seed: int
    num_queries: int
    config_path: str
    model_registry_path: str
    weights_source: str = ""
    resolved_utility_weights: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
