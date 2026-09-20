"""
BlissBot - retrieval evaluation harness (works with ANY retriever).

Usage as a script (sanity baseline, no API keys needed):
    python 03_eval_harness.py bliss_corpus_clean.json golden_eval_set.json

Usage with your own retriever (Step 3+):
    from eval_harness import load, evaluate
    corpus, golden = load("bliss_corpus_clean.json", "golden_eval_set.json")
    def my_retrieve(query, k, history=None, include_tier_b=False):   # -> list[doc_id] best-first
        ...
    evaluate(my_retrieve, corpus, golden, k_list=(1, 3, 5, 10))
"""
import json, math, sys, collections
from pathlib import Path
corpus_path = Path('out_dir/bliss_corpus_clean.json')
golden_path = Path('out_dir/golden_eval_set.json')

def load(corpus_path, golden_path):
    return json.load(open(corpus_path, encoding="utf-8")), json.load(open(golden_path, encoding="utf-8"))

# ----------------------------------------------------------------------------- metrics
def hit_at_k(ranked, gold, k, min_grade=2):
    rel = {g["doc_id"] for g in gold if g["grade"] >= min_grade} or {g["doc_id"] for g in gold}
    return float(any(d in rel for d in ranked[:k]))

def recall_at_k(ranked, gold, k):
    rel = {g["doc_id"] for g in gold}
    return len(rel & set(ranked[:k])) / len(rel) if rel else 0.0

def mrr(ranked, gold, min_grade=2, cutoff=10):
    rel = {g["doc_id"] for g in gold if g["grade"] >= min_grade} or {g["doc_id"] for g in gold}
    for i, d in enumerate(ranked[:cutoff], 1):
        if d in rel: return 1.0 / i
    return 0.0

def ndcg_at_k(ranked, gold, k):
    gains = {g["doc_id"]: g["grade"] for g in gold}
    dcg = sum(gains.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0

def qgroup_hit(ranked, qgroups, doc_to_qgroup, k):
    return float(any(doc_to_qgroup.get(d) in qgroups for d in ranked[:k]))

# ----------------------------------------------------------------------------- runner
def evaluate(retrieve, corpus, golden, k_list=(1, 3, 5, 10), verbose=True, followup_mode="standalone"):
    """retrieve(query, k, history=None, include_tier_b=False) -> ranked list of doc_ids.
    followup_mode: 'standalone' uses the ideal rewrite; 'raw' passes the raw follow-up (+history) to your retriever."""
    d2g = {d["doc_id"]: d["qgroup"] for d in corpus}
    K = max(k_list); rows = []
    for q in golden["queries"]:
        if q["expected_route"] not in ("rag", "rag_sensitive", "rag_low_confidence"): continue
        if not q["gold"] and not q.get("gold_qgroups"): continue
        query, hist = q["query"], q.get("history")
        if q["category"] == "followup" and followup_mode == "standalone": query, hist = q["standalone_query"], None
        inc_b = q["category"] == "tier_b_only"
        ranked = retrieve(query, K, history=hist, include_tier_b=inc_b)
        r = {"qid": q["qid"], "category": q["category"], "query": q["query"], "ranked": ranked[:K]}
        if q["gold"]:
            for k in k_list:
                r[f"hit@{k}"] = hit_at_k(ranked, q["gold"], k); r[f"recall@{k}"] = recall_at_k(ranked, q["gold"], k)
                r[f"ndcg@{k}"] = ndcg_at_k(ranked, q["gold"], k)
            r["mrr"] = mrr(ranked, q["gold"], cutoff=K)
        if q.get("gold_qgroups"):
            r["qgroup_hit@5"] = qgroup_hit(ranked, q["gold_qgroups"], d2g, 5)
        rows.append(r)
    if verbose: report(rows, k_list)
    return rows

def report(rows, k_list):
    cols = [f"hit@{k}" for k in k_list] + [f"recall@{max(k_list)}", f"ndcg@{k_list[len(k_list)//2]}", "mrr"]
    def agg(rs):
        out = {}
        for c in cols:
            v = [r[c] for r in rs if c in r]; out[c] = sum(v) / len(v) if v else float("nan")
        return out
    print(f"{'category':14} {'n':>4} " + " ".join(f"{c:>9}" for c in cols))
    by = collections.defaultdict(list)
    for r in rows: by[r["category"]].append(r)
    for cat, rs in [("ALL", rows)] + sorted(by.items()):
        a = agg(rs); n = sum("mrr" in r for r in rs); print(f"{cat:14} {n:>4} " + " ".join(f"{a[c]:>9.3f}" for c in cols))
    qg = [r["qgroup_hit@5"] for r in rows if "qgroup_hit@5" in r]
    if qg: print(f"tier-B question-group hit@5: {sum(qg)/len(qg):.3f} (n={len(qg)})")

# ----------------------------------------------------------------------------- sanity baseline
def tfidf_retriever(corpus, text_fn):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    A = [d for d in corpus if d["in_index"] and d["tier"] == "A"]
    AB = [d for d in corpus if d["in_index"]]
    def build(docs):
        v = TfidfVectorizer(sublinear_tf=True, stop_words="english", ngram_range=(1, 2), min_df=1)
        return v, v.fit_transform([text_fn(d) for d in docs]), [d["doc_id"] for d in docs]
    idx = {False: build(A), True: build(AB)}
    def retrieve(query, k, history=None, include_tier_b=False):
        v, M, ids = idx[include_tier_b]
        s = (M @ v.transform([query]).T).toarray().ravel()
        return [ids[i] for i in np.argsort(-s)[:k]]
    return retrieve

if __name__ == "__main__":
    corpus, golden = load(corpus_path, golden_path)
    for name, fn in [("TF-IDF on question only", lambda d: d["question"]),
                     ("TF-IDF on question + answer", lambda d: d["question"] + " " + d["answer"])]:
        print(f"\n=== sanity baseline: {name} (tier A index; tier A+B for tier_b_only queries) ===")
        rows = evaluate(tfidf_retriever(corpus, fn), corpus, golden)
        miss = [r for r in rows if "hit@10" in r and r["hit@10"] == 0]
        print(f"\nqueries with NO direct-answer doc in top-10 ({len(miss)}) -> inspect these for label errors or real retrieval gaps:")
        for r in miss: print(f"  {r['qid']} [{r['category']}] {r['query']}")
