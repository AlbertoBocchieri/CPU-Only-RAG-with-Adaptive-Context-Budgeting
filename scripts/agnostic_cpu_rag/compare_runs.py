from __future__ import annotations
import json
import sys
from pathlib import Path

def load_json(path: str):
    return json.loads(Path(path).read_text())

def load_jsonl(path: str):
    rows = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[row['qid']] = row
    return rows

base_summary = load_json(sys.argv[1])
new_summary = load_json(sys.argv[2])
base_rows = load_jsonl(sys.argv[3])
new_rows = load_jsonl(sys.argv[4])
qids = [q for q in base_rows if q in new_rows]

def metric(summary, section, key, sub=None):
    node = summary.get(section, {})
    if sub is None:
        return node.get(key)
    return node.get(key, {}).get(sub)

out = {
    'base': {
        'relevant_doc_recall': metric(base_summary,'metrics_mean','relevant_doc_recall'),
        'pair_in_context': metric(base_summary,'metrics_mean','pair_in_context'),
        'context_mean': metric(base_summary,'latency_summary_ms','context_tokens_used','mean'),
        't_total_p50': metric(base_summary,'latency_summary_ms','t_total_ms','p50'),
    },
    'new': {
        'relevant_doc_recall': metric(new_summary,'metrics_mean','relevant_doc_recall'),
        'pair_in_context': metric(new_summary,'metrics_mean','pair_in_context'),
        'context_mean': metric(new_summary,'latency_summary_ms','context_tokens_used','mean'),
        't_total_p50': metric(new_summary,'latency_summary_ms','t_total_ms','p50'),
    }
}
out['delta'] = {k: (out['new'][k] or 0) - (out['base'][k] or 0) for k in out['base']}
recall_better = recall_worse = pair_better = pair_worse = ctx_lower = ctx_higher = 0
for q in qids:
    b = base_rows[q]
    n = new_rows[q]
    br = b.get('metrics', {}).get('relevant_doc_recall')
    nr = n.get('metrics', {}).get('relevant_doc_recall')
    bp = b.get('metrics', {}).get('pair_in_context')
    np = n.get('metrics', {}).get('pair_in_context')
    bc = b.get('context_controller', {}).get('context_tokens_used') or 0
    nc = n.get('context_controller', {}).get('context_tokens_used') or 0
    if nr is not None and br is not None:
        recall_better += nr > br
        recall_worse += nr < br
    if np is not None and bp is not None:
        pair_better += np > bp
        pair_worse += np < bp
    ctx_lower += nc < bc
    ctx_higher += nc > bc
out['paired'] = {
    'num_qids': len(qids),
    'recall_better': int(recall_better),
    'recall_worse': int(recall_worse),
    'pair_better': int(pair_better),
    'pair_worse': int(pair_worse),
    'context_lower': int(ctx_lower),
    'context_higher': int(ctx_higher),
}
print(json.dumps(out, indent=2))
