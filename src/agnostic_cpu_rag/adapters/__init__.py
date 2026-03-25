from .base import DatasetAdapter, TaskAdapter
from .datasets import (
    BEIRDatasetAdapter,
    HotpotDatasetAdapter,
    NaturalQuestionsDatasetAdapter,
    TwoWikiDatasetAdapter,
    make_dataset_adapter,
)
from .tasks import (
    MultiHopQATaskAdapter,
    OpenQATaskAdapter,
    RetrievalOnlyTaskAdapter,
    make_task_adapter,
)

__all__ = [
    "BEIRDatasetAdapter",
    "DatasetAdapter",
    "HotpotDatasetAdapter",
    "MultiHopQATaskAdapter",
    "NaturalQuestionsDatasetAdapter",
    "OpenQATaskAdapter",
    "RetrievalOnlyTaskAdapter",
    "TaskAdapter",
    "TwoWikiDatasetAdapter",
    "make_dataset_adapter",
    "make_task_adapter",
]
