#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

from agnostic_cpu_rag.weight_search import answer_length_bucket, question_prefix_bucket


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_hotpot_rows() -> list[dict[str, Any]]:
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    rows: list[dict[str, Any]] = []
    for row in ds:
        qid = f"hotpot_{row['id']}"
        qtype = str(row.get("type", "")).strip() or "unknown"
        level = str(row.get("level", "")).strip() or "unknown"
        rows.append(
            {
                "qid": qid,
                "question": str(row.get("question", "")).strip(),
                "stratum": f"{qtype}|{level}",
                "type": qtype,
                "level": level,
            }
        )
    return rows


def load_twowiki_rows() -> list[dict[str, Any]]:
    ds = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    rows: list[dict[str, Any]] = []
    for row in ds:
        qid = f"2wiki_{row['id']}"
        qtype = str(row.get("type", "")).strip() or "unknown"
        rows.append(
            {
                "qid": qid,
                "question": str(row.get("question", "")).strip(),
                "stratum": qtype,
                "type": qtype,
            }
        )
    return rows


def load_squad_rows() -> list[dict[str, Any]]:
    ds = load_dataset("squad", split="validation")
    rows: list[dict[str, Any]] = []
    for row in ds:
        qid = f"squad_{row['id']}"
        answers = [str(a).strip() for a in row.get("answers", {}).get("text", []) if str(a).strip()]
        prefix = question_prefix_bucket(str(row.get("question", "")))
        length_bucket = answer_length_bucket(answers)
        rows.append(
            {
                "qid": qid,
                "question": str(row.get("question", "")).strip(),
                "stratum": f"{prefix}|{length_bucket}",
                "question_prefix_bucket": prefix,
                "answer_length_bucket": length_bucket,
            }
        )
    return rows


def stratified_take(rows: list[dict[str, Any]], *, size: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if size < 0:
        raise ValueError("size must be non-negative")
    if size > len(rows):
        raise ValueError(f"Requested {size} rows from pool of size {len(rows)}")
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["stratum"])].append(row)
    for key in sorted(buckets):
        rng.shuffle(buckets[key])
    ordered_keys = sorted(buckets)
    selected: list[dict[str, Any]] = []
    while len(selected) < size:
        progressed = False
        for key in ordered_keys:
            bucket = buckets[key]
            if not bucket:
                continue
            selected.append(bucket.pop())
            progressed = True
            if len(selected) >= size:
                break
        if not progressed:
            break
    selected_qids = {row["qid"] for row in selected}
    remaining = [row for row in rows if row["qid"] not in selected_qids]
    return selected, remaining


def summarize_distribution(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for key in keys:
        ctr = Counter(str(row.get(key, "")) for row in rows)
        out[key] = dict(sorted(ctr.items()))
    return out


def proportional_targets(full_rows: list[dict[str, Any]], size: int, key: str) -> dict[str, int]:
    total = len(full_rows)
    counts = Counter(str(row.get(key, "")) for row in full_rows)
    targets = {label: int(round(size * (count / max(1, total)))) for label, count in counts.items()}
    shortfall = size - sum(targets.values())
    if shortfall != 0:
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        idx = 0
        while shortfall != 0 and ranked:
            label = ranked[idx % len(ranked)][0]
            if shortfall > 0:
                targets[label] = targets.get(label, 0) + 1
                shortfall -= 1
            elif targets.get(label, 0) > 0:
                targets[label] -= 1
                shortfall += 1
            idx += 1
    return dict(sorted(targets.items()))


def emit_pool(
    *,
    out_dir: Path,
    dataset_name: str,
    pool_name: str,
    rows: list[dict[str, Any]],
    full_rows: list[dict[str, Any]],
    distribution_keys: list[str],
    notes: dict[str, Any] | None = None,
) -> None:
    qids = [row["qid"] for row in rows]
    payload = {
        "dataset": dataset_name,
        "pool_name": pool_name,
        "size": len(rows),
        "qids_path": str(out_dir / f"{pool_name}_qids.json"),
        "distribution_target": {key: proportional_targets(full_rows, len(rows), key) for key in distribution_keys},
        "distribution_actual": summarize_distribution(rows, distribution_keys),
        "notes": dict(notes or {}),
    }
    atomic_write_json(out_dir / f"{pool_name}_qids.json", qids)
    atomic_write_json(out_dir / f"{pool_name}_meta.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical disjoint pools for task-family utility-weight search.")
    parser.add_argument("--output-root", default="results/task_family_weight_search/pools")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hotpot-representative-qids",
        default="results/retrieval_quality_pool300/cfg_a000fd2bb017/hotpot_qa/sampled_qids.json",
    )
    args = parser.parse_args()

    out_root = Path(args.output_root)
    seed = int(args.seed)

    # Hotpot
    hotpot_rows = load_hotpot_rows()
    hotpot_by_qid = {row["qid"]: row for row in hotpot_rows}
    hotpot_rep_qids = json.loads(Path(args.hotpot_representative_qids).read_text(encoding="utf-8"))
    hotpot_rep_rows = [hotpot_by_qid[qid] for qid in hotpot_rep_qids if qid in hotpot_by_qid]
    hotpot_remaining = [row for row in hotpot_rows if row["qid"] not in set(hotpot_rep_qids)]
    hotpot_rep_150, _ = stratified_take(hotpot_rep_rows, size=150, seed=seed)
    hotpot_tuning_300, hotpot_remaining = stratified_take(hotpot_remaining, size=300, seed=seed + 1)
    hotpot_tuning_150, _ = stratified_take(hotpot_tuning_300, size=150, seed=seed + 2)
    hotpot_pilot_75, hotpot_remaining = stratified_take(hotpot_remaining, size=75, seed=seed + 3)
    hotpot_holdout, _ = stratified_take(hotpot_remaining, size=1000, seed=seed + 4)
    hotpot_dir = out_root / "hotpot_qa"
    emit_pool(
        out_dir=hotpot_dir,
        dataset_name="hotpot_qa",
        pool_name="representative_150",
        rows=hotpot_rep_150,
        full_rows=hotpot_rows,
        distribution_keys=["stratum", "type", "level"],
        notes={
            "pre_established": True,
            "non_blind": True,
            "subset_of": "representative_300",
            "source_qids_path": str(args.hotpot_representative_qids),
        },
    )
    emit_pool(
        out_dir=hotpot_dir,
        dataset_name="hotpot_qa",
        pool_name="representative_300",
        rows=hotpot_rep_rows,
        full_rows=hotpot_rows,
        distribution_keys=["stratum", "type", "level"],
        notes={
            "pre_established": True,
            "non_blind": True,
            "source_qids_path": str(args.hotpot_representative_qids),
        },
    )
    emit_pool(
        out_dir=hotpot_dir,
        dataset_name="hotpot_qa",
        pool_name="tuning_150",
        rows=hotpot_tuning_150,
        full_rows=hotpot_rows,
        distribution_keys=["stratum", "type", "level"],
        notes={"subset_of": "tuning_300", "disjoint_from": ["representative_150", "representative_300"]},
    )
    emit_pool(
        out_dir=hotpot_dir,
        dataset_name="hotpot_qa",
        pool_name="tuning_300",
        rows=hotpot_tuning_300,
        full_rows=hotpot_rows,
        distribution_keys=["stratum", "type", "level"],
        notes={"disjoint_from": ["representative_300"]},
    )
    emit_pool(
        out_dir=hotpot_dir,
        dataset_name="hotpot_qa",
        pool_name="holdout_1000",
        rows=hotpot_holdout,
        full_rows=hotpot_rows,
        distribution_keys=["stratum", "type", "level"],
        notes={"disjoint_from": ["representative_300", "tuning_300", "selection_pilot_75"]},
    )
    emit_pool(
        out_dir=hotpot_dir,
        dataset_name="hotpot_qa",
        pool_name="selection_pilot_75",
        rows=hotpot_pilot_75,
        full_rows=hotpot_rows,
        distribution_keys=["stratum", "type", "level"],
        notes={"disjoint_from": ["representative_300", "tuning_300", "holdout_1000"]},
    )
    atomic_write_json(
        hotpot_dir / "pool_manifest.json",
        {
            "dataset": "hotpot_qa",
            "split_size": len(hotpot_rows),
            "seed": seed,
            "note": "Hotpot representative validation was performed on a pre-established canonical pool, not sampled post-hoc from the final validation split.",
            "pools": {
                "tuning_150": "disjoint_from representative_300, selection_pilot_75, holdout_1000",
                "tuning_300": "disjoint_from representative_300, selection_pilot_75, holdout_1000",
                "representative_150": "subset_of representative_300",
                "representative_300": "pre_established_canonical_non_blind",
                "selection_pilot_75": "disjoint_from tuning_300, representative_300, holdout_1000",
                "holdout_1000": "disjoint_from tuning_300, representative_300, selection_pilot_75",
            },
        },
    )

    # 2Wiki
    twowiki_rows = load_twowiki_rows()
    twowiki_dir = out_root / "two_wiki_multihop"
    twowiki_tuning_300, twowiki_remaining = stratified_take(twowiki_rows, size=300, seed=seed)
    twowiki_tuning_150, _ = stratified_take(twowiki_tuning_300, size=150, seed=seed + 1)
    twowiki_rep_300, twowiki_remaining = stratified_take(twowiki_remaining, size=300, seed=seed + 2)
    twowiki_rep_150, _ = stratified_take(twowiki_rep_300, size=150, seed=seed + 3)
    twowiki_pilot_75, twowiki_remaining = stratified_take(twowiki_remaining, size=75, seed=seed + 4)
    twowiki_holdout, _ = stratified_take(twowiki_remaining, size=1000, seed=seed + 5)
    for pool_name, rows in (
        ("tuning_150", twowiki_tuning_150),
        ("tuning_300", twowiki_tuning_300),
        ("representative_150", twowiki_rep_150),
        ("representative_300", twowiki_rep_300),
        ("selection_pilot_75", twowiki_pilot_75),
        ("holdout_1000", twowiki_holdout),
    ):
        emit_pool(
            out_dir=twowiki_dir,
            dataset_name="two_wiki_multihop",
            pool_name=pool_name,
            rows=rows,
            full_rows=twowiki_rows,
            distribution_keys=["stratum", "type"],
        )
    atomic_write_json(
        twowiki_dir / "pool_manifest.json",
        {
            "dataset": "two_wiki_multihop",
            "split_size": len(twowiki_rows),
            "seed": seed,
            "type_distribution": summarize_distribution(twowiki_rows, ["type"])["type"],
            "pools": {
                "tuning_150": "subset_of tuning_300",
                "tuning_300": "disjoint_from representative_300, selection_pilot_75, holdout_1000",
                "representative_150": "subset_of representative_300",
                "representative_300": "disjoint_from tuning_300, selection_pilot_75, holdout_1000",
                "selection_pilot_75": "disjoint_from tuning_300, representative_300, holdout_1000",
                "holdout_1000": "disjoint_from tuning_300, representative_300, selection_pilot_75",
            },
        },
    )

    # SQuAD-open
    squad_rows = load_squad_rows()
    squad_dir = out_root / "squad_open"
    squad_tuning_300, squad_remaining = stratified_take(squad_rows, size=300, seed=seed)
    squad_tuning_150, _ = stratified_take(squad_tuning_300, size=150, seed=seed + 1)
    squad_rep_300, squad_remaining = stratified_take(squad_remaining, size=300, seed=seed + 2)
    squad_rep_150, _ = stratified_take(squad_rep_300, size=150, seed=seed + 3)
    squad_pilot_75, squad_remaining = stratified_take(squad_remaining, size=75, seed=seed + 4)
    squad_holdout, _ = stratified_take(squad_remaining, size=1000, seed=seed + 5)
    for pool_name, rows in (
        ("tuning_150", squad_tuning_150),
        ("tuning_300", squad_tuning_300),
        ("representative_150", squad_rep_150),
        ("representative_300", squad_rep_300),
        ("selection_pilot_75", squad_pilot_75),
        ("holdout_1000", squad_holdout),
    ):
        emit_pool(
            out_dir=squad_dir,
            dataset_name="squad_open",
            pool_name=pool_name,
            rows=rows,
            full_rows=squad_rows,
            distribution_keys=["stratum", "question_prefix_bucket", "answer_length_bucket"],
        )
    atomic_write_json(
        squad_dir / "pool_manifest.json",
        {
            "dataset": "squad_open",
            "split_size": len(squad_rows),
            "seed": seed,
            "question_prefix_distribution": summarize_distribution(squad_rows, ["question_prefix_bucket"])["question_prefix_bucket"],
            "answer_length_distribution": summarize_distribution(squad_rows, ["answer_length_bucket"])["answer_length_bucket"],
            "pools": {
                "tuning_150": "subset_of tuning_300",
                "tuning_300": "disjoint_from representative_300, selection_pilot_75, holdout_1000",
                "representative_150": "subset_of representative_300",
                "representative_300": "disjoint_from tuning_300, selection_pilot_75, holdout_1000",
                "selection_pilot_75": "disjoint_from tuning_300, representative_300, holdout_1000",
                "holdout_1000": "disjoint_from tuning_300, representative_300, selection_pilot_75",
            },
        },
    )


if __name__ == "__main__":
    main()
