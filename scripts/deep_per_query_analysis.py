#!/usr/bin/env python3
"""Deep per-query analysis of final full runs: Q0 vs ACB-SC for HotpotQA and 2WikiMultihop."""

import json
import numpy as np
from collections import Counter, defaultdict
from scipy import stats

# ── File paths ──────────────────────────────────────────────────────────────
FILES = {
    "HotpotQA": {
        "q0": "/Users/albertobocchieri/Desktop/rag_from_scratch/results/final_full_runs/hotpot_q0_full7405/cfg_a000fd2bb017/hotpot_qa/per_query.jsonl",
        "acbsc": "/Users/albertobocchieri/Desktop/rag_from_scratch/results/final_full_runs/hotpot_acbsc_full7405/cfg_32831c771548/hotpot_qa/per_query.jsonl",
    },
    "2WikiMultihop": {
        "q0": "/Users/albertobocchieri/Desktop/rag_from_scratch/results/final_full_runs/twowiki_q0_large6288/cfg_65250e4d919f/two_wiki_multihop/per_query.jsonl",
        "acbsc": "/Users/albertobocchieri/Desktop/rag_from_scratch/results/final_full_runs/twowiki_acbsc_large6288/cfg_da394a03c05b/two_wiki_multihop/per_query.jsonl",
    },
}


def load_jsonl(path):
    records = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            records[obj["qid"]] = obj
    return records


def extract(rec):
    """Pull flat fields from a record."""
    am = rec["answer_metrics_per_query"]
    lat = rec["latency_ms"]
    cb = rec.get("context_budgeting", {})
    pc = rec.get("post_context", {})
    res = rec.get("resources", {})
    tok = rec.get("tokens", {})
    return {
        "em": am["em"],
        "f1": am["f1"],
        "ttft_ms": lat.get("ttft_ms"),
        "t_total_ms": lat.get("t_total_ms"),
        "t_prefill_ms": lat.get("t_prefill_ms"),
        "t_llm_total_ms": lat.get("t_llm_total_ms"),
        "context_tokens_used": cb.get("context_tokens_used"),
        "k_eff": cb.get("k_eff"),
        "pair_in_context": pc.get("pair_in_context_at_k"),
        "power_mean_watts": res.get("power_mean_watts"),
        "power_samples": res.get("power_samples"),
        "power_sampling_interval_ms": res.get("power_sampling_interval_ms"),
        "budget_cap_tokens": cb.get("budget_cap_tokens"),
        "budget_cap_source": cb.get("budget_cap_source"),
        "ewma_prefill_ms_per_token": cb.get("ewma_prefill_ms_per_token"),
        "query_index": rec.get("query_index"),
        "context_tokens": tok.get("context_tokens"),
    }


def pct(arr, p):
    return np.percentile(arr, p)


def bootstrap_ci(data, n_boot=10000, seed=42, ci=0.95):
    rng = np.random.RandomState(seed)
    means = np.empty(n_boot)
    n = len(data)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        means[i] = np.mean(data[idx])
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return lo, hi


def mcnemar_test(em_q0, em_acbsc):
    """McNemar test on paired EM."""
    a = np.array(em_q0)
    b = np.array(em_acbsc)
    # b correct, a wrong  vs  a correct, b wrong
    b_correct_a_wrong = np.sum((b == 1) & (a == 0))
    a_correct_b_wrong = np.sum((a == 1) & (b == 0))
    # McNemar chi-sq (no continuity correction)
    n_disc = b_correct_a_wrong + a_correct_b_wrong
    if n_disc == 0:
        return 0.0, 1.0
    chi2 = (b_correct_a_wrong - a_correct_b_wrong) ** 2 / n_disc
    p_val = 1 - stats.chi2.cdf(chi2, df=1)
    return chi2, p_val, b_correct_a_wrong, a_correct_b_wrong


SEP = "=" * 80


def analyze_dataset(name, q0_path, acbsc_path):
    print(f"\n{SEP}")
    print(f"  DATASET: {name}")
    print(SEP)

    q0_raw = load_jsonl(q0_path)
    acbsc_raw = load_jsonl(acbsc_path)

    # Pair by qid
    common_qids = sorted(set(q0_raw.keys()) & set(acbsc_raw.keys()))
    print(f"\nTotal Q0 queries: {len(q0_raw)}")
    print(f"Total ACB-SC queries: {len(acbsc_raw)}")
    print(f"Paired queries: {len(common_qids)}")
    q0_only = set(q0_raw.keys()) - set(acbsc_raw.keys())
    acbsc_only = set(acbsc_raw.keys()) - set(q0_raw.keys())
    if q0_only:
        print(f"  Q0-only: {len(q0_only)}")
    if acbsc_only:
        print(f"  ACB-SC-only: {len(acbsc_only)}")

    # Extract paired data
    q0_data = [extract(q0_raw[qid]) for qid in common_qids]
    acbsc_data = [extract(acbsc_raw[qid]) for qid in common_qids]
    N = len(common_qids)

    # Convenience arrays
    def arr(data, key):
        return np.array([d[key] for d in data if d[key] is not None], dtype=float)

    def paired_arr(key):
        vals_q0, vals_ac = [], []
        for i in range(N):
            v0 = q0_data[i][key]
            va = acbsc_data[i][key]
            if v0 is not None and va is not None:
                vals_q0.append(v0)
                vals_ac.append(va)
        return np.array(vals_q0), np.array(vals_ac)

    # ── A. Quality Deltas ───────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  A. QUALITY DELTAS")
    print(f"{'─'*60}")

    f1_q0, f1_ac = paired_arr("f1")
    em_q0, em_ac = paired_arr("em")
    delta_f1 = f1_ac - f1_q0
    delta_em = em_ac - em_q0

    print(f"\n  N paired = {len(delta_f1)}")
    print(f"  Mean F1:  Q0 = {np.mean(f1_q0):.4f},  ACB-SC = {np.mean(f1_ac):.4f}")
    print(f"  Mean EM:  Q0 = {np.mean(em_q0):.4f},  ACB-SC = {np.mean(em_ac):.4f}")
    print(f"  Mean ΔF1  (ACB-SC − Q0) = {np.mean(delta_f1):+.4f}")
    print(f"  Mean ΔEM  (ACB-SC − Q0) = {np.mean(delta_em):+.4f}")

    lo_f1, hi_f1 = bootstrap_ci(delta_f1)
    lo_em, hi_em = bootstrap_ci(delta_em)
    print(f"  Bootstrap 95% CI on ΔF1: [{lo_f1:+.4f}, {hi_f1:+.4f}]")
    print(f"  Bootstrap 95% CI on ΔEM: [{lo_em:+.4f}, {hi_em:+.4f}]")

    chi2, p_val, b_corr_a_wrong, a_corr_b_wrong = mcnemar_test(em_q0, em_ac)
    print(f"\n  McNemar test (EM):")
    print(f"    ACB-SC correct & Q0 wrong: {b_corr_a_wrong}")
    print(f"    Q0 correct & ACB-SC wrong: {a_corr_b_wrong}")
    print(f"    chi² = {chi2:.4f},  p = {p_val:.6f}")

    better_f1 = np.sum(delta_f1 > 1e-9)
    worse_f1 = np.sum(delta_f1 < -1e-9)
    tied_f1 = np.sum(np.abs(delta_f1) <= 1e-9)
    print(f"\n  F1 wins/losses/ties: ACB-SC better={better_f1}, worse={worse_f1}, tied={tied_f1}")

    better_em = np.sum(delta_em > 1e-9)
    worse_em = np.sum(delta_em < -1e-9)
    tied_em = np.sum(np.abs(delta_em) <= 1e-9)
    print(f"  EM wins/losses/ties: ACB-SC better={better_em}, worse={worse_em}, tied={tied_em}")

    # ── B. Context Analysis ─────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  B. CONTEXT ANALYSIS")
    print(f"{'─'*60}")

    ctx_q0 = arr(q0_data, "context_tokens_used")
    ctx_ac = arr(acbsc_data, "context_tokens_used")
    for label, a in [("Q0", ctx_q0), ("ACB-SC", ctx_ac)]:
        print(f"\n  context_tokens_used ({label}): n={len(a)}")
        print(f"    mean={np.mean(a):.1f}  median={np.median(a):.1f}")
        print(f"    p5={pct(a,5):.1f}  p25={pct(a,25):.1f}  p75={pct(a,75):.1f}  p95={pct(a,95):.1f}")

    keff_q0 = arr(q0_data, "k_eff")
    keff_ac = arr(acbsc_data, "k_eff")
    for label, a in [("Q0", keff_q0), ("ACB-SC", keff_ac)]:
        print(f"\n  k_eff ({label}): mean={np.mean(a):.2f}  median={np.median(a):.1f}")

    # k_eff histogram for ACB-SC
    print(f"\n  k_eff distribution (ACB-SC):")
    keff_int = keff_ac.astype(int)
    for k in range(1, 11):
        c = np.sum(keff_int == k)
        print(f"    k={k:2d}: {c:5d}  ({100*c/len(keff_int):.1f}%)")
    c11_15 = np.sum((keff_int >= 11) & (keff_int <= 15))
    c16_20 = np.sum((keff_int >= 16) & (keff_int <= 20))
    c_gt20 = np.sum(keff_int > 20)
    print(f"    k=11-15: {c11_15:5d}  ({100*c11_15/len(keff_int):.1f}%)")
    print(f"    k=16-20: {c16_20:5d}  ({100*c16_20/len(keff_int):.1f}%)")
    if c_gt20 > 0:
        print(f"    k>20:    {c_gt20:5d}  ({100*c_gt20/len(keff_int):.1f}%)")

    # ── C. Latency Analysis ─────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  C. LATENCY ANALYSIS")
    print(f"{'─'*60}")

    for metric_key, metric_label in [("ttft_ms", "TTFT (ms)"), ("t_total_ms", "t_total (ms)")]:
        print(f"\n  {metric_label}:")
        v_q0 = arr(q0_data, metric_key)
        v_ac = arr(acbsc_data, metric_key)
        for label, a in [("Q0", v_q0), ("ACB-SC", v_ac)]:
            print(f"    {label:8s}: mean={np.mean(a):8.1f}  p50={pct(a,50):8.1f}  p75={pct(a,75):8.1f}  p95={pct(a,95):8.1f}  p99={pct(a,99):8.1f}")

    # Speedup ratios
    ttft_q0 = arr(q0_data, "ttft_ms")
    ttft_ac = arr(acbsc_data, "ttft_ms")
    tt_q0 = arr(q0_data, "t_total_ms")
    tt_ac = arr(acbsc_data, "t_total_ms")
    print(f"\n  Speedup (Q0/ACB-SC):")
    print(f"    TTFT   p50: {pct(ttft_q0,50)/pct(ttft_ac,50):.2f}x   p95: {pct(ttft_q0,95)/pct(ttft_ac,95):.2f}x")
    print(f"    t_total p50: {pct(tt_q0,50)/pct(tt_ac,50):.2f}x   p95: {pct(tt_q0,95)/pct(tt_ac,95):.2f}x")

    # ── D. Coverage Analysis ────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  D. COVERAGE ANALYSIS (pair_in_context)")
    print(f"{'─'*60}")

    pic_q0_arr = arr(q0_data, "pair_in_context")
    pic_ac_arr = arr(acbsc_data, "pair_in_context")
    print(f"\n  pair_in_context mean:  Q0 = {np.mean(pic_q0_arr):.4f},  ACB-SC = {np.mean(pic_ac_arr):.4f}")

    # Transition matrix
    pic_q0_list = []
    pic_ac_list = []
    f1_q0_pic = []
    f1_ac_pic = []
    for i in range(N):
        pq = q0_data[i].get("pair_in_context")
        pa = acbsc_data[i].get("pair_in_context")
        if pq is not None and pa is not None:
            pic_q0_list.append(pq)
            pic_ac_list.append(pa)
            f1_q0_pic.append(q0_data[i]["f1"])
            f1_ac_pic.append(acbsc_data[i]["f1"])

    pic_q0_b = np.array(pic_q0_list)
    pic_ac_b = np.array(pic_ac_list)
    f1_q0_pic = np.array(f1_q0_pic)
    f1_ac_pic = np.array(f1_ac_pic)

    both_1 = np.sum((pic_q0_b == 1) & (pic_ac_b == 1))
    both_0 = np.sum((pic_q0_b == 0) & (pic_ac_b == 0))
    lost = np.sum((pic_q0_b == 1) & (pic_ac_b == 0))  # Q0 had, ACB-SC lost
    gained = np.sum((pic_q0_b == 0) & (pic_ac_b == 1))  # Q0 didn't, ACB-SC gained

    print(f"\n  Transition matrix (Q0 -> ACB-SC):")
    print(f"    Both have pair:    {both_1:5d}")
    print(f"    Neither has pair:  {both_0:5d}")
    print(f"    ACB-SC LOST pair:  {lost:5d}")
    print(f"    ACB-SC GAINED pair:{gained:5d}")

    # F1 analysis for lost/kept
    mask_lost = (pic_q0_b == 1) & (pic_ac_b == 0)
    if np.sum(mask_lost) > 0:
        print(f"\n  Queries where ACB-SC LOST pair_in_context (n={np.sum(mask_lost)}):")
        print(f"    Mean F1 Q0:     {np.mean(f1_q0_pic[mask_lost]):.4f}")
        print(f"    Mean F1 ACB-SC: {np.mean(f1_ac_pic[mask_lost]):.4f}")
        print(f"    ΔF1:            {np.mean(f1_ac_pic[mask_lost]) - np.mean(f1_q0_pic[mask_lost]):+.4f}")

    mask_kept = (pic_q0_b == 1) & (pic_ac_b == 1)
    if np.sum(mask_kept) > 0:
        print(f"\n  Queries where ACB-SC KEPT pair_in_context (n={np.sum(mask_kept)}):")
        print(f"    Mean F1 Q0:     {np.mean(f1_q0_pic[mask_kept]):.4f}")
        print(f"    Mean F1 ACB-SC: {np.mean(f1_ac_pic[mask_kept]):.4f}")
        print(f"    ΔF1:            {np.mean(f1_ac_pic[mask_kept]) - np.mean(f1_q0_pic[mask_kept]):+.4f}")

    mask_gained = (pic_q0_b == 0) & (pic_ac_b == 1)
    if np.sum(mask_gained) > 0:
        print(f"\n  Queries where ACB-SC GAINED pair_in_context (n={np.sum(mask_gained)}):")
        print(f"    Mean F1 Q0:     {np.mean(f1_q0_pic[mask_gained]):.4f}")
        print(f"    Mean F1 ACB-SC: {np.mean(f1_ac_pic[mask_gained]):.4f}")
        print(f"    ΔF1:            {np.mean(f1_ac_pic[mask_gained]) - np.mean(f1_q0_pic[mask_gained]):+.4f}")

    # ── E. Error Analysis ───────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  E. ERROR ANALYSIS")
    print(f"{'─'*60}")

    # Queries where ACB-SC is much worse
    ctx_q0_full, ctx_ac_full = paired_arr("context_tokens_used")
    pic_q0_full, pic_ac_full = paired_arr("pair_in_context")

    mask_big_loss = delta_f1 < -0.1
    mask_big_gain = delta_f1 > 0.1
    n_loss = np.sum(mask_big_loss)
    n_gain = np.sum(mask_big_gain)

    print(f"\n  Queries where ACB-SC F1 < Q0 F1 by > 0.1: {n_loss}")
    if n_loss > 0:
        # Need to align indices - delta_f1 is based on paired_arr for f1
        # Use the same index set
        ctx_q0_all = np.array([q0_data[i]["context_tokens_used"] for i in range(N)])
        ctx_ac_all = np.array([acbsc_data[i]["context_tokens_used"] for i in range(N)])
        pic_q0_all = np.array([q0_data[i]["pair_in_context"] if q0_data[i]["pair_in_context"] is not None else np.nan for i in range(N)])
        pic_ac_all = np.array([acbsc_data[i]["pair_in_context"] if acbsc_data[i]["pair_in_context"] is not None else np.nan for i in range(N)])

        # delta_f1 length should equal N
        print(f"    Mean context_tokens_used ACB-SC: {np.mean(ctx_ac_all[mask_big_loss]):.1f}")
        print(f"    Mean context_tokens_used Q0:     {np.mean(ctx_q0_all[mask_big_loss]):.1f}")
        print(f"    Mean pair_in_context Q0:         {np.nanmean(pic_q0_all[mask_big_loss]):.4f}")
        print(f"    Mean pair_in_context ACB-SC:     {np.nanmean(pic_ac_all[mask_big_loss]):.4f}")
        print(f"    Mean ΔF1:                        {np.mean(delta_f1[mask_big_loss]):+.4f}")
    else:
        ctx_q0_all = np.array([q0_data[i]["context_tokens_used"] for i in range(N)])
        ctx_ac_all = np.array([acbsc_data[i]["context_tokens_used"] for i in range(N)])
        pic_q0_all = np.array([q0_data[i]["pair_in_context"] if q0_data[i]["pair_in_context"] is not None else np.nan for i in range(N)])
        pic_ac_all = np.array([acbsc_data[i]["pair_in_context"] if acbsc_data[i]["pair_in_context"] is not None else np.nan for i in range(N)])

    print(f"\n  Queries where ACB-SC F1 > Q0 F1 by > 0.1: {n_gain}")
    if n_gain > 0:
        print(f"    Mean context_tokens_used ACB-SC: {np.mean(ctx_ac_all[mask_big_gain]):.1f}")
        print(f"    Mean context_tokens_used Q0:     {np.mean(ctx_q0_all[mask_big_gain]):.1f}")
        print(f"    Mean pair_in_context Q0:         {np.nanmean(pic_q0_all[mask_big_gain]):.4f}")
        print(f"    Mean pair_in_context ACB-SC:     {np.nanmean(pic_ac_all[mask_big_gain]):.4f}")
        print(f"    Mean ΔF1:                        {np.mean(delta_f1[mask_big_gain]):+.4f}")

    # Correlations
    delta_ctx = ctx_ac_all - ctx_q0_all
    r_pearson, p_pearson = stats.pearsonr(delta_f1, delta_ctx)
    r_spearman, p_spearman = stats.spearmanr(delta_f1, delta_ctx)
    print(f"\n  Correlation ΔF1 vs Δcontext_tokens:")
    print(f"    Pearson  r = {r_pearson:+.4f},  p = {p_pearson:.6f}")
    print(f"    Spearman ρ = {r_spearman:+.4f},  p = {p_spearman:.6f}")

    # ── F. EWMA / Budget Analysis (ACB-SC only) ────────────────────────────
    print(f"\n{'─'*60}")
    print("  F. EWMA / BUDGET ANALYSIS (ACB-SC only)")
    print(f"{'─'*60}")

    budget_caps = np.array([acbsc_data[i]["budget_cap_tokens"] for i in range(N)
                            if acbsc_data[i]["budget_cap_tokens"] is not None], dtype=float)
    if len(budget_caps) > 0:
        print(f"\n  budget_cap_tokens: mean={np.mean(budget_caps):.1f}  std={np.std(budget_caps):.1f}  min={np.min(budget_caps):.0f}  max={np.max(budget_caps):.0f}")

    # budget_cap_source breakdown
    src_counter = Counter()
    for i in range(N):
        src = acbsc_data[i].get("budget_cap_source")
        if src is not None:
            src_counter[src] += 1
    print(f"\n  budget_cap_source breakdown:")
    for src, cnt in src_counter.most_common():
        print(f"    {src}: {cnt} ({100*cnt/N:.1f}%)")

    # EWMA prefill ms per token (post-warmup)
    ewma_vals = []
    for i in range(N):
        qi = acbsc_data[i].get("query_index", i)
        val = acbsc_data[i].get("ewma_prefill_ms_per_token")
        if val is not None and qi is not None and qi >= 8:
            ewma_vals.append(val)
    ewma_vals = np.array(ewma_vals)
    if len(ewma_vals) > 0:
        print(f"\n  ewma_prefill_ms_per_token (post-warmup, n={len(ewma_vals)}):")
        print(f"    mean={np.mean(ewma_vals):.4f}  std={np.std(ewma_vals):.4f}  min={np.min(ewma_vals):.4f}  max={np.max(ewma_vals):.4f}")
    else:
        print(f"\n  ewma_prefill_ms_per_token: no post-warmup values found (all None or < 8 queries)")

    # ── G. Energy Analysis ──────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  G. ENERGY ANALYSIS")
    print(f"{'─'*60}")

    def compute_energy(data_list):
        """Energy per query in Joules: power_mean_watts * (t_total_ms / 1000)"""
        energies = []
        for d in data_list:
            pw = d.get("power_mean_watts")
            tt = d.get("t_total_ms")
            if pw is not None and tt is not None and pw > 0:
                energies.append(pw * tt / 1000.0)  # Joules
        return np.array(energies)

    e_q0 = compute_energy(q0_data)
    e_ac = compute_energy(acbsc_data)

    if len(e_q0) > 0 and len(e_ac) > 0:
        print(f"\n  Energy per query (Joules):")
        print(f"    Q0:     mean={np.mean(e_q0):.2f}  median={np.median(e_q0):.2f}  total={np.sum(e_q0):.1f} J ({np.sum(e_q0)/3600:.2f} Wh)")
        print(f"    ACB-SC: mean={np.mean(e_ac):.2f}  median={np.median(e_ac):.2f}  total={np.sum(e_ac):.1f} J ({np.sum(e_ac)/3600:.2f} Wh)")
        print(f"    Ratio (Q0/ACB-SC): mean={np.mean(e_q0)/np.mean(e_ac):.2f}x  total={np.sum(e_q0)/np.sum(e_ac):.2f}x")
    else:
        print(f"\n  No power/energy data available.")

    # Check for queries with power_status issues
    power_ok_q0 = sum(1 for qid in common_qids if q0_raw[qid].get("resources", {}).get("power_status") == "ok")
    power_ok_ac = sum(1 for qid in common_qids if acbsc_raw[qid].get("resources", {}).get("power_status") == "ok")
    print(f"\n  Power status OK: Q0={power_ok_q0}/{N}, ACB-SC={power_ok_ac}/{N}")


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for ds_name, paths in FILES.items():
        analyze_dataset(ds_name, paths["q0"], paths["acbsc"])
    print(f"\n{SEP}")
    print("  ANALYSIS COMPLETE")
    print(SEP)
