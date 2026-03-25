from __future__ import annotations

import hashlib
import random
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset

from .types import Document, QAExample


def _stable_context_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def load_squad_qa(
    train_size: int,
    val_size: int,
    test_size: int,
    max_corpus_docs: int,
    seed: int,
) -> tuple[dict[str, list[QAExample]], dict[str, Document]]:
    rng = random.Random(seed)
    ds = load_dataset("squad")

    train_pool = list(range(len(ds["train"])))
    val_pool = list(range(len(ds["validation"])))

    rng.shuffle(train_pool)
    rng.shuffle(val_pool)

    train_take = train_pool[:train_size]
    val_take = train_pool[train_size : train_size + val_size]
    test_take = val_pool[:test_size]

    def to_examples(split_name: str, idxs: list[int]) -> list[QAExample]:
        src = ds[split_name]
        out: list[QAExample] = []
        for i in idxs:
            row = src[i]
            doc_id = f"squad_{_stable_context_id(row['context'])}"
            answers = [a.strip() for a in row["answers"]["text"] if a.strip()]
            if not answers:
                continue
            out.append(
                QAExample(
                    qid=f"{split_name}_{row['id']}",
                    question=row["question"].strip(),
                    answers=answers,
                    gold_doc_ids={doc_id},
                )
            )
        return out

    splits = {
        "train": to_examples("train", train_take),
        "val": to_examples("train", val_take),
        "test": to_examples("validation", test_take),
    }

    context_counter: Counter[str] = Counter()
    context_by_doc: dict[str, str] = {}

    for split_name, idxs in (("train", train_take), ("train", val_take), ("validation", test_take)):
        src = ds[split_name]
        for i in idxs:
            context = src[i]["context"]
            doc_id = f"squad_{_stable_context_id(context)}"
            context_counter[doc_id] += 1
            context_by_doc[doc_id] = context

    kept_doc_ids = [d for d, _ in context_counter.most_common(max_corpus_docs)]
    kept_set = set(kept_doc_ids)

    for split_name in list(splits):
        splits[split_name] = [
            ex for ex in splits[split_name] if any(doc_id in kept_set for doc_id in ex.gold_doc_ids)
        ]

    corpus = {
        doc_id: Document(doc_id=doc_id, text=context_by_doc[doc_id], title="SQuAD Context")
        for doc_id in kept_doc_ids
    }

    return splits, corpus


def load_beir_dataset(
    dataset_name: str,
    max_queries: int,
    seed: int,
    data_root: str = "data/beir",
) -> tuple[dict[str, Document], dict[str, str], dict[str, dict[str, int]]]:
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    rng = random.Random(seed)
    data_root_path = Path(data_root)
    data_root_path.mkdir(parents=True, exist_ok=True)

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = util.download_and_unzip(url, str(data_root_path))

    corpus_raw, queries_raw, qrels = GenericDataLoader(data_path).load(split="test")

    qids = [qid for qid in queries_raw if qid in qrels and len(qrels[qid]) > 0]
    rng.shuffle(qids)
    qids = qids[:max_queries]

    queries = {qid: queries_raw[qid] for qid in qids}
    filtered_qrels: dict[str, dict[str, int]] = {}

    relevant_doc_ids: set[str] = set()
    for qid in qids:
        rel = {doc_id: int(v) for doc_id, v in qrels[qid].items() if int(v) > 0}
        if not rel:
            continue
        filtered_qrels[qid] = rel
        relevant_doc_ids.update(rel.keys())

    # Keep full corpus for realistic retrieval evaluation; all ids are still included.
    docs = {
        doc_id: Document(
            doc_id=doc_id,
            title=(item.get("title") or "").strip(),
            text=((item.get("title") or "") + "\n" + (item.get("text") or "")).strip(),
            metadata={"dataset": dataset_name},
        )
        for doc_id, item in corpus_raw.items()
    }

    return docs, queries, filtered_qrels


def stratified_sample_examples(
    examples: list[QAExample],
    max_samples: int,
    seed: int,
    strata_key: str = "strata",
) -> list[QAExample]:
    if max_samples <= 0 or len(examples) <= max_samples:
        return examples

    rng = random.Random(seed)
    buckets: dict[str, list[QAExample]] = {}
    for ex in examples:
        key = str(ex.metadata.get(strata_key, "default"))
        buckets.setdefault(key, []).append(ex)

    bucket_keys = sorted(buckets.keys())
    for key in bucket_keys:
        rng.shuffle(buckets[key])

    out: list[QAExample] = []
    while len(out) < max_samples:
        progressed = False
        for key in bucket_keys:
            if buckets[key]:
                out.append(buckets[key].pop())
                progressed = True
                if len(out) >= max_samples:
                    break
        if not progressed:
            break
    return out


def _doc_id_from_title(prefix: str, title: str) -> str:
    norm_title = " ".join(title.lower().split())
    return f"{prefix}_{_stable_context_id(norm_title)}"


def load_hotpotqa_distractor(
    split: str,
    max_queries: int,
    seed: int,
    include_qids: list[str] | None = None,
) -> tuple[list[QAExample], dict[str, Document]]:
    ds = load_dataset("hotpot_qa", "distractor", split=split)
    docs: dict[str, Document] = {}
    examples: list[QAExample] = []

    for row in ds:
        qid = f"hotpot_{row['id']}"
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        if not question or not answer:
            continue

        titles = list(row["context"]["title"])
        sentences = list(row["context"]["sentences"])
        title_to_doc_id: dict[str, str] = {}

        for title, sent_list in zip(titles, sentences, strict=False):
            text = " ".join([str(s).strip() for s in sent_list if str(s).strip()]).strip()
            if not text:
                continue
            doc_id = _doc_id_from_title("hotpot", str(title))
            title_to_doc_id[str(title)] = doc_id
            prev = docs.get(doc_id)
            if prev is None or len(text) > len(prev.text):
                docs[doc_id] = Document(
                    doc_id=doc_id,
                    title=str(title),
                    text=text,
                    metadata={"dataset": "hotpot_qa"},
                )

        support_titles = [str(t) for t in row["supporting_facts"]["title"]]
        support_sent_ids = [int(x) for x in row["supporting_facts"]["sent_id"]]
        gold_doc_ids = {title_to_doc_id[t] for t in support_titles if t in title_to_doc_id}
        if not gold_doc_ids:
            continue
        context_doc_ids: list[str] = []
        seen_doc_ids: set[str] = set()
        for title in titles:
            doc_id = title_to_doc_id.get(str(title))
            if not doc_id or doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            context_doc_ids.append(doc_id)

        examples.append(
            QAExample(
                qid=qid,
                question=question,
                answers=[answer],
                gold_doc_ids=gold_doc_ids,
                metadata={
                    "dataset": "hotpot_qa",
                    "type": str(row.get("type", "")),
                    "level": str(row.get("level", "")),
                    "strata": f"{row.get('type', '')}|{row.get('level', '')}",
                    "supporting_facts": {
                        "title": support_titles,
                        "sent_id": support_sent_ids,
                    },
                    "context_doc_ids": context_doc_ids,
                },
            )
        )

    if include_qids:
        by_qid = {ex.qid: ex for ex in examples}
        ordered: list[QAExample] = []
        for qid in include_qids:
            ex = by_qid.get(str(qid))
            if ex is not None:
                ordered.append(ex)
        examples = ordered
    else:
        examples = stratified_sample_examples(examples, max_samples=max_queries, seed=seed, strata_key="strata")

    # Keep corpus bounded to sampled queries (all context docs for each sampled query).
    keep_doc_ids: set[str] = set()
    for ex in examples:
        keep_doc_ids.update(ex.gold_doc_ids)
        context_ids = ex.metadata.get("context_doc_ids", []) if ex.metadata else []
        keep_doc_ids.update([str(doc_id) for doc_id in context_ids if str(doc_id)])
    docs = {doc_id: doc for doc_id, doc in docs.items() if doc_id in keep_doc_ids}
    return examples, docs


def load_two_wiki_multihop_validation(
    max_queries: int,
    seed: int,
    include_qids: list[str] | None = None,
) -> tuple[list[QAExample], dict[str, Document]]:
    ds = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    docs: dict[str, Document] = {}
    examples: list[QAExample] = []

    for row in ds:
        qid = f"2wiki_{row['id']}"
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        if not question or not answer:
            continue

        context = row.get("context", {})
        titles = list(context.get("title", []))
        sentences = list(context.get("sentences", []))
        title_to_doc_id: dict[str, str] = {}

        for title, sent_list in zip(titles, sentences, strict=False):
            text = " ".join([str(s).strip() for s in sent_list if str(s).strip()]).strip()
            if not text:
                continue
            doc_id = _doc_id_from_title("2wiki", str(title))
            title_to_doc_id[str(title)] = doc_id
            prev = docs.get(doc_id)
            if prev is None or len(text) > len(prev.text):
                docs[doc_id] = Document(
                    doc_id=doc_id,
                    title=str(title),
                    text=text,
                    metadata={"dataset": "two_wiki_multihop"},
                )

        supporting_facts = row.get("supporting_facts", {})
        support_titles = [str(t) for t in supporting_facts.get("title", [])]
        support_sent_ids = [int(x) for x in supporting_facts.get("sent_id", [])]
        gold_doc_ids = {title_to_doc_id[t] for t in support_titles if t in title_to_doc_id}
        if not gold_doc_ids:
            continue

        context_doc_ids: list[str] = []
        seen_doc_ids: set[str] = set()
        for title in titles:
            doc_id = title_to_doc_id.get(str(title))
            if not doc_id or doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            context_doc_ids.append(doc_id)

        examples.append(
            QAExample(
                qid=qid,
                question=question,
                answers=[answer],
                gold_doc_ids=gold_doc_ids,
                metadata={
                    "dataset": "two_wiki_multihop",
                    "type": str(row.get("type", "")),
                    "strata": str(row.get("type", "")),
                    "evidences": row.get("evidences", []),
                    "supporting_facts": {
                        "title": support_titles,
                        "sent_id": support_sent_ids,
                    },
                    "context_doc_ids": context_doc_ids,
                },
            )
        )

    if include_qids:
        by_qid = {ex.qid: ex for ex in examples}
        ordered: list[QAExample] = []
        for qid in include_qids:
            ex = by_qid.get(str(qid))
            if ex is not None:
                ordered.append(ex)
        examples = ordered
    else:
        examples = stratified_sample_examples(examples, max_samples=max_queries, seed=seed, strata_key="strata")

    keep_doc_ids: set[str] = set()
    for ex in examples:
        keep_doc_ids.update(ex.gold_doc_ids)
        context_ids = ex.metadata.get("context_doc_ids", []) if ex.metadata else []
        keep_doc_ids.update([str(doc_id) for doc_id in context_ids if str(doc_id)])
    docs = {doc_id: doc for doc_id, doc in docs.items() if doc_id in keep_doc_ids}
    return examples, docs


def _clean_nq_doc_text(tokens: dict[str, Any]) -> str:
    words = list(tokens.get("token", []))
    is_html = list(tokens.get("is_html", []))
    clean = []
    for tok, html in zip(words, is_html, strict=False):
        if html:
            continue
        t = str(tok).strip()
        if t:
            clean.append(t)
    return " ".join(clean).strip()


def _extract_nq_answers(row: dict[str, Any]) -> tuple[list[str], str]:
    ann = row.get("annotations", {})
    out: list[str] = []
    answer_type = "none"

    for sa in ann.get("short_answers", []):
        for text in sa.get("text", []):
            v = str(text).strip()
            if v:
                out.append(v)
    if out:
        return out, "short"

    for yn in ann.get("yes_no_answer", []):
        if int(yn) == 0:
            return ["no"], "yes_no"
        if int(yn) == 1:
            return ["yes"], "yes_no"

    long_answers = ann.get("long_answer", [])
    toks = row.get("document", {}).get("tokens", {})
    words = list(toks.get("token", []))
    is_html = list(toks.get("is_html", []))
    for la in long_answers:
        start = int(la.get("start_token", -1))
        end = int(la.get("end_token", -1))
        if start < 0 or end <= start or end > len(words):
            continue
        span = []
        for tok, html in zip(words[start:end], is_html[start:end], strict=False):
            if html:
                continue
            t = str(tok).strip()
            if t:
                span.append(t)
        text = " ".join(span).strip()
        if text:
            out.append(text)
            answer_type = "long"
            break

    return out, answer_type


def _convert_nq_row(row: dict[str, Any]) -> tuple[QAExample, Document] | None:
    qid = f"nq_{row['id']}"
    question = str(row["question"]["text"]).strip()
    if not question:
        return None
    doc = row.get("document", {})
    title = str(doc.get("title", "")).strip()
    url = str(doc.get("url", "")).strip()
    text = _clean_nq_doc_text(doc.get("tokens", {}))
    if not text:
        return None
    doc_key = f"{title}|{url}" if (title or url) else text[:120]
    doc_id = f"nqdoc_{_stable_context_id(doc_key)}"

    answers, answer_type = _extract_nq_answers(row)
    answers = [a for a in answers if a.strip()]
    if not answers:
        return None

    return (
        QAExample(
            qid=qid,
            question=question,
            answers=answers,
            gold_doc_ids={doc_id},
            metadata={
                "dataset": "natural_questions",
                "answer_type": answer_type,
                "strata": answer_type,
                "doc_id": doc_id,
                "doc_title": title,
            },
        ),
        Document(
            doc_id=doc_id,
            title=title,
            text=text,
            metadata={"dataset": "natural_questions", "url": url},
        ),
    )


def load_natural_questions_validation(
    max_queries: int,
    seed: int,
    include_qids: list[str] | None = None,
    streaming: bool = False,
) -> tuple[list[QAExample], dict[str, Document]]:
    ds = load_dataset("natural_questions", split="validation", streaming=bool(streaming))
    docs: dict[str, Document] = {}
    examples: list[QAExample] = []
    requested_qids = [str(qid) for qid in include_qids] if include_qids else None
    requested_set = set(requested_qids) if requested_qids else None

    for row in ds:
        converted = _convert_nq_row(row)
        if converted is None:
            continue
        example, document = converted
        if requested_set is not None and example.qid not in requested_set:
            continue
        docs[document.doc_id] = document
        examples.append(example)
        if requested_set is None and streaming and len(examples) >= int(max_queries):
            break
        if requested_set is not None and len({ex.qid for ex in examples}) >= len(requested_set):
            break

    if requested_qids:
        by_qid = {ex.qid: ex for ex in examples}
        ordered: list[QAExample] = []
        for qid in requested_qids:
            ex = by_qid.get(str(qid))
            if ex is not None:
                ordered.append(ex)
        examples = ordered
    elif not streaming:
        examples = stratified_sample_examples(examples, max_samples=max_queries, seed=seed, strata_key="strata")

    keep_doc_ids: set[str] = set()
    for ex in examples:
        keep_doc_ids.update(ex.gold_doc_ids)
    docs = {doc_id: doc for doc_id, doc in docs.items() if doc_id in keep_doc_ids}
    return examples, docs


def load_squad_open_validation(
    max_queries: int,
    seed: int,
    include_qids: list[str] | None = None,
) -> tuple[list[QAExample], dict[str, Document]]:
    ds = load_dataset("squad", split="validation")
    docs: dict[str, Document] = {}
    examples: list[QAExample] = []

    for row in ds:
        question = str(row.get("question", "")).strip()
        title = str(row.get("title", "")).strip()
        context = str(row.get("context", "")).strip()
        answers = [str(a).strip() for a in row.get("answers", {}).get("text", []) if str(a).strip()]
        if not question or not context or not answers:
            continue

        qid = f"squad_{row['id']}"
        doc_key = f"{title}|{context[:200]}"
        doc_id = f"squaddoc_{_stable_context_id(doc_key)}"
        docs[doc_id] = Document(
            doc_id=doc_id,
            title=title,
            text=context,
            metadata={"dataset": "squad_open"},
        )
        examples.append(
            QAExample(
                qid=qid,
                question=question,
                answers=answers,
                gold_doc_ids={doc_id},
                metadata={
                    "dataset": "squad_open",
                    "title": title,
                    "strata": title or "default",
                    "doc_id": doc_id,
                },
            )
        )

    if include_qids:
        by_qid = {ex.qid: ex for ex in examples}
        ordered: list[QAExample] = []
        for qid in include_qids:
            ex = by_qid.get(str(qid))
            if ex is not None:
                ordered.append(ex)
        examples = ordered
    else:
        rng = random.Random(seed)
        rng.shuffle(examples)
        examples = examples[:max_queries]

    keep_doc_ids: set[str] = set()
    for ex in examples:
        keep_doc_ids.update(ex.gold_doc_ids)
    docs = {doc_id: doc for doc_id, doc in docs.items() if doc_id in keep_doc_ids}
    return examples, docs


def build_qrels_from_qa_examples(examples: list[QAExample]) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for ex in examples:
        qrels[ex.qid] = {doc_id: 1 for doc_id in ex.gold_doc_ids}
    return qrels


def map_doc_qrels_to_chunk_qrels(
    doc_qrels: dict[str, dict[str, int]],
    doc_to_chunks: dict[str, list[str]],
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for qid, rel_docs in doc_qrels.items():
        rel_chunks: dict[str, int] = {}
        for doc_id, grade in rel_docs.items():
            for chunk_id in doc_to_chunks.get(doc_id, []):
                rel_chunks[chunk_id] = max(rel_chunks.get(chunk_id, 0), grade)
        out[qid] = rel_chunks
    return out
