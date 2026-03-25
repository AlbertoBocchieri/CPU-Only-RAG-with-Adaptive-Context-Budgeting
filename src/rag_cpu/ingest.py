from __future__ import annotations

from pathlib import Path

from .types import Document


SUPPORTED_EXT = {".txt", ".md", ".rst"}


def load_local_documents(path: str | Path) -> dict[str, Document]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Corpus path not found: {root}")

    docs: dict[str, Document] = {}
    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in SUPPORTED_EXT:
            continue
        text = f.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        doc_id = str(f.relative_to(root)).replace("/", "__")
        docs[doc_id] = Document(doc_id=doc_id, text=text, title=f.name, metadata={"path": str(f)})

    return docs
