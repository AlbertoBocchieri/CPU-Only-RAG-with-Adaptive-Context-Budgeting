from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..records import CoverageGoal, DatasetBundle, GoldReference


class TaskAdapter(ABC):
    name: str
    family: str
    coverage_goal: CoverageGoal
    supports_generation: bool = True

    @abstractmethod
    def required_distinct_docs(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def evaluate_answer(self, prediction: str, gold: GoldReference) -> dict[str, float | None]:
        raise NotImplementedError

    @abstractmethod
    def evaluate_context(self, selected_doc_ids: list[str], gold: GoldReference) -> dict[str, float | int | None]:
        raise NotImplementedError


class DatasetAdapter(ABC):
    name: str
    default_split: str
    task_family_hint: str

    @abstractmethod
    def load(
        self,
        *,
        max_queries: int,
        seed: int,
        include_qids: list[str] | None = None,
        split: str | None = None,
        **kwargs: Any,
    ) -> DatasetBundle:
        raise NotImplementedError
