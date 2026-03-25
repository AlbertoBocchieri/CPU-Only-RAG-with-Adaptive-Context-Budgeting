from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    doc_id: str
    text: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QAExample:
    qid: str
    question: str
    answers: list[str]
    gold_doc_ids: set[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedItem:
    item_id: str
    text: str
    score: float
    doc_id: str
    source: str
