from __future__ import annotations

from typing import Any


def answer_template_for_task(task_family: str) -> str:
    family = str(task_family).strip().lower()
    if family == "retrieval_only":
        return ""
    return (
        "Context:\n"
        "{context}\n\n"
        "Answer the question using only the context above. "
        "Return only the final answer string, with no explanation.\n\n"
        "Q: {question}\n"
        "A:"
    )


def answer_sanity(answer: str) -> dict[str, Any]:
    text = str(answer or "").strip()
    tokens = text.split()
    malformed = int(not text)
    too_long = int(len(tokens) > 24)
    return {
        "empty": int(not text),
        "too_long": too_long,
        "malformed": malformed,
        "token_count": len(tokens),
        "looks_valid": bool(text) and len(tokens) <= 24,
    }
