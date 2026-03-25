from __future__ import annotations

from .types import Chunk, Document


def _word_chunks(words: list[str], chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    step = max(1, chunk_size - overlap)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        spans.append((start, end))
        if end == len(words):
            break
        start += step
    return spans


def chunk_documents(
    docs: dict[str, Document],
    chunk_size_words: int,
    chunk_overlap_words: int,
    min_chunk_words: int,
) -> tuple[dict[str, Chunk], dict[str, list[str]]]:
    chunks: dict[str, Chunk] = {}
    doc_to_chunks: dict[str, list[str]] = {}

    for doc_id, doc in docs.items():
        words = doc.text.split()
        if not words:
            continue

        spans = _word_chunks(words, chunk_size_words, chunk_overlap_words)
        doc_chunk_ids: list[str] = []

        for idx, (start, end) in enumerate(spans):
            span_words = words[start:end]
            if len(span_words) < min_chunk_words and idx != 0:
                continue
            chunk_text = " ".join(span_words).strip()
            if not chunk_text:
                continue
            chunk_id = f"{doc_id}::c{idx:04d}"
            chunks[chunk_id] = Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=chunk_text,
                metadata={"start_word": start, "end_word": end},
            )
            doc_chunk_ids.append(chunk_id)

        if doc_chunk_ids:
            doc_to_chunks[doc_id] = doc_chunk_ids

    return chunks, doc_to_chunks
