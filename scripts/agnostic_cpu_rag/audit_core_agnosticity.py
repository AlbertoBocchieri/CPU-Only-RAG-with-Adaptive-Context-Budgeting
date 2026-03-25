#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


BANNED_TERMS = (
    "hotpot",
    "two_wiki",
    "2wiki",
    "squad",
    "scifact",
    "natural_questions",
    "written",
    "directed",
    "produced",
    "american",
    "british",
    "french",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the agnostic core for benchmark-shaped lexical terms.")
    parser.add_argument("--root", default="src/agnostic_cpu_rag")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    findings: list[dict[str, object]] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.as_posix()
        if "/adapters/" in rel or rel.endswith("/adapters.py"):
            continue
        text = path.read_text(encoding="utf-8")
        lower = text.lower()
        hits = [term for term in BANNED_TERMS if term in lower]
        if hits:
            findings.append({"path": rel, "hits": hits})

    out = {
        "root": str(root),
        "banned_terms": list(BANNED_TERMS),
        "num_files_with_hits": len(findings),
        "findings": findings,
    }
    payload = json.dumps(out, indent=2, ensure_ascii=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
