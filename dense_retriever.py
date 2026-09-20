"""
BlissBot - Step 3a: dense (embedding) retrieval, compared against the TF-IDF baseline.

What it does
  1. Embeds the tier-A docs (optionally tier B) with one or more open embedding models.
  2. Tries several ways of representing a Q&A doc:
        q        embed the question only
        a        embed the answer only
        qa       embed "question + answer" as one text
        q_max_a  score = max(sim(query, question), sim(query, answer))
        q_avg_a  score = mean(sim(query, question), sim(query, answer))
  3. Scores every (model, variant) on the golden set with the same harness as the TF-IDF baseline.
  4. Prints a comparison table, and shows which queries got FIXED / BROKEN versus TF-IDF.

Setup (once):   pip install sentence-transformers
Run:            python 04_dense_retriever.py                       # default: 3 small models
                python 04_dense_retriever.py --models bge-small     # one model
                python 04_dense_retriever.py --models fake-hash     # smoke test, no download
                python 04_dense_retriever.py --tier-b               # also embed CounselChat rows (slower)

Embeddings are cached in ./cache so re-runs are fast.
CHECK each model's card on huggingface.co for the recommended query/passage prefixes before trusting results.
"""
import argparse, csv, hashlib, importlib.util, json, sys, time, zlib
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path("/workspaces/BlissBot")

# ----------------------------------------------------------------------------- model registry
# q_prefix / d_prefix are model-specific instructions. Wrong prefixes silently hurt quality.
MODELS = {
    "minilm":     dict(name="sentence-transformers/all-MiniLM-L6-v2", q_prefix="", d_prefix="", max_len=256),

    "bge-small":  dict(name="BAAI/bge-small-en-v1.5",
                       q_prefix="Represent this sentence for searching relevant passages: ", d_prefix="", max_len=512),

    "bge-base":   dict(name="BAAI/bge-base-en-v1.5",
                       q_prefix="Represent this sentence for searching relevant passages: ", d_prefix="", max_len=512),

    "e5-base":    dict(name="intfloat/e5-base-v2", q_prefix="query: ", d_prefix="passage: ", max_len=512),

    # heavier (0.6B params). needs recent sentence-transformers + transformers; uses the model's built-in "query" prompt
    "qwen3-0.6b": dict(name="Qwen/Qwen3-Embedding-0.6B", query_prompt_name="query", max_len=512),

    #Clinical / Domain-Specific: BioLinkBERT
    "bio-link-bert": dict(name="michiyasunaga/BioLinkBERT-large", q_prefix="query: ", d_prefix="passage:", max_len=512),

    # smoke-test embedder: hashed word/char n-grams. NOT semantic. Only proves the pipeline runs.
    "fake-hash":  dict(name="fake-hash", max_len=0),
}
DEFAULT_MODELS = 'bge-small' #"minilm,bge-small,e5-base,bio-link-bert"
ALL_VARIANTS = ["q", "a", "qa", "q_max_a", "q_avg_a"]
DEFAULT_VARIANTS = "q_max_a" 
FIELDS_FOR = {"q": {"q"}, "a": {"a"}, "qa": {"qa"}, "q_max_a": {"q", "a"}, "q_avg_a": {"q", "a"}}

def load_harness():
    for p in (PROJECT_ROOT / "eval_harness.py", PROJECT_ROOT / "03_eval_harness.py"):
        if p.exists():
            spec = importlib.util.spec_from_file_location("eval_harness", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    sys.exit(f"Could not find eval_harness.py or 03_eval_harness.py in {PROJECT_ROOT}.")


def load_tfidf_retriever():
    p = PROJECT_ROOT / "tf-idf_retriever.py"
    if not p.exists():
        sys.exit(f"Could not find tf-idf_retriever.py in {PROJECT_ROOT}.")
    spec = importlib.util.spec_from_file_location("tfidf_module", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ----------------------------------------------------------------------------- embedder
def fake_encode(texts, dim=384):
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        t = t.lower(); toks = t.split() + [t[j:j + 3] for j in range(max(1, len(t) - 2))]
        for tok in toks: out[i, zlib.crc32(tok.encode()) % dim] += 1.0
    n = np.linalg.norm(out, axis=1, keepdims=True); n[n == 0] = 1
    return out / n

class Embedder:
    def __init__(self, key, device="cpu"):
        self.key, self.cfg = key, MODELS[key]
        if self.cfg["name"] == "fake-hash": self.model = None; return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            sys.exit("Run:  pip install sentence-transformers")
        print(f"  loading {self.cfg['name']} (first run downloads the weights)...")
        self.model = SentenceTransformer(self.cfg["name"], device=device)
        if self.cfg.get("max_len"): self.model.max_seq_length = min(self.cfg["max_len"], 512)

    def encode(self, texts, kind, batch_size=32):
        if self.model is None: return fake_encode(texts)
        kw = dict(batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                  show_progress_bar=len(texts) > 300)
        if kind == "query":
            if self.cfg.get("query_prompt_name"): kw["prompt_name"] = self.cfg["query_prompt_name"]
            texts = [self.cfg.get("q_prefix", "") + t for t in texts]
        else:
            texts = [self.cfg.get("d_prefix", "") + t for t in texts]
        return self.model.encode(texts, **kw).astype(np.float32)

def doc_text(d, field):
    return {"q": d["question"], "a": d["answer"], "qa": d["question"] + "\n" + d["answer"]}[field]

def embed_docs(emb, docs, field, cache_dir, batch_size):
    cfg = emb.cfg
    sig = json.dumps([cfg["name"], cfg.get("q_prefix"), cfg.get("d_prefix"), cfg.get("max_len"), field,
                      [(d["doc_id"], doc_text(d, field)) for d in docs]], ensure_ascii=False)
    h = hashlib.sha1(sig.encode()).hexdigest()[:12]
    path = Path(cache_dir) / f"{emb.key}_{field}_{h}.npy"
    if path.exists(): return np.load(path), 0.0, True
    t0 = time.time(); M = emb.encode([doc_text(d, field) for d in docs], "doc", batch_size)
    path.parent.mkdir(parents=True, exist_ok=True); np.save(path, M)
    return M, time.time() - t0, False

# ----------------------------------------------------------------------------- retriever
def make_retriever(emb, docs, mats, variant, qcache):
    ids = np.array([d["doc_id"] for d in docs]); is_a = np.array([d["tier"] == "A" for d in docs])
    def qvec(q):
        if q not in qcache: qcache[q] = emb.encode([q], "query")[0]
        return qcache[q]
    def retrieve(query, k, history=None, include_tier_b=False):
        v = qvec(query)
        if variant in ("q", "a", "qa"): s = mats[variant] @ v
        else:
            sq, sa = mats["q"] @ v, mats["a"] @ v
            s = np.maximum(sq, sa) if variant == "q_max_a" else (sq + sa) / 2
        if not include_tier_b: s = np.where(is_a, s, -np.inf)
        return ids[np.argsort(-s)[:k]].tolist()
    return retrieve

# ----------------------------------------------------------------------------- summarising
def mean(rows, key):
    v = [r[key] for r in rows if key in r]; return sum(v) / len(v) if v else float("nan")

def summarise(rows):
    by = {}
    for r in rows: by.setdefault(r["category"], []).append(r)
    out = {"n": sum("mrr" in r for r in rows), "hit@1": mean(rows, "hit@1"), "hit@5": mean(rows, "hit@5"),
           "hit@10": mean(rows, "hit@10"), "ndcg@5": mean(rows, "ndcg@5"), "mrr": mean(rows, "mrr")}
    for cat in ("colloquial", "paraphrase", "exact_term", "followup", "hard_negative"):
        out[f"{cat}_hit@5"] = mean(by.get(cat, []), "hit@5")
    return out

def print_table(results):
    cols = ["hit@1", "hit@5", "hit@10", "ndcg@5", "mrr", "colloquial_hit@5", "paraphrase_hit@5", "exact_term_hit@5", "followup_hit@5"]
    heads = ["Hit@1", "Hit@5", "Hit@10", "nDCG@5", "MRR", "coll@5", "para@5", "exact@5", "follow@5"]
    print(f"\n{'config':28}" + "".join(f"{h:>9}" for h in heads))
    for name, s in sorted(results.items(), key=lambda kv: -kv[1]["mrr"]):
        print(f"{name:28}" + "".join(f"{s[c]:>9.3f}" for c in cols))

def by_qid(rows): return {r["qid"]: r for r in rows}

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(PROJECT_ROOT / "out_dir" / "bliss_corpus_clean.json"))
    ap.add_argument("--golden", default=str(PROJECT_ROOT / "out_dir" / "golden_eval_set.json"))
    ap.add_argument("--models", default=DEFAULT_MODELS, help=f"comma list from: {', '.join(MODELS)}")
    ap.add_argument("--variants", default=DEFAULT_VARIANTS)
    ap.add_argument("--tier-b", action="store_true", help="also embed CounselChat rows (slower); enables tier_b_only scoring")
    ap.add_argument("--cache-dir", default="cache"); ap.add_argument("--results-dir", default="results")
    ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    eh = load_harness()
    tfidf_mod = load_tfidf_retriever()
    corpus, golden = eh.load(a.corpus, a.golden)
    docs = [d for d in corpus if d["in_index"] and (d["tier"] == "A" or a.tier_b)]
    print(f"Indexing {len(docs)} docs ({'tier A + B' if a.tier_b else 'tier A only'})")
    if not a.tier_b:   # question-group-only gold can't be scored without tier B docs
        golden = dict(golden); golden["queries"] = [{k: v for k, v in q.items() if k != "gold_qgroups"} for q in golden["queries"]]
    variants = a.variants.split(",")

    results, all_rows = {}, {}
    for name, fn in [("tfidf | q", lambda d: d["question"]), ("tfidf | qa", lambda d: d["question"] + " " + d["answer"])]:
        rows = eh.evaluate(tfidf_mod.tfidf_retriever(corpus, fn), corpus, golden, verbose=False)
        results[name] = summarise(rows); all_rows[name] = rows

    for mk in a.models.split(","):
        if mk not in MODELS: sys.exit(f"unknown model {mk}; choose from {list(MODELS)}")
        print(f"\n== {mk} ==")
        emb = Embedder(mk, a.device)
        need = set().union(*(FIELDS_FOR[v] for v in variants))
        mats = {}
        for f in sorted(need):
            M, secs, cached = embed_docs(emb, docs, f, a.cache_dir, a.batch_size)
            mats[f] = M; print(f"  docs[{f}] {M.shape} {'(cache)' if cached else f'{secs:.0f}s'}")
        qcache = {}
        for v in variants:
            rows = eh.evaluate(make_retriever(emb, docs, mats, v, qcache), corpus, golden, verbose=False)
            results[f"{mk} | {v}"] = summarise(rows); all_rows[f"{mk} | {v}"] = rows

    print_table(results)

    Path(a.results_dir).mkdir(exist_ok=True)
    with open(Path(a.results_dir) / "dense_summary.csv", "w", newline="") as f:
        w = csv.writer(f); cols = list(next(iter(results.values())).keys()); w.writerow(["config"] + cols)
        for k, s in results.items(): w.writerow([k] + [round(s[c], 4) if isinstance(s[c], float) else s[c] for c in cols])

    # what changed vs the best TF-IDF config?
    dense = {k: v for k, v in results.items() if not k.startswith("tfidf")}
    if dense:
        best = max(dense, key=lambda k: dense[k]["mrr"])
        base = max((k for k in results if k.startswith("tfidf")), key=lambda k: results[k]["mrr"])
        B, T = by_qid(all_rows[best]), by_qid(all_rows[base])
        fixed = [q for q in B if T[q].get("hit@10") == 0 and B[q].get("hit@5") == 1]
        broke = [q for q in B if T[q].get("hit@5") == 1 and B[q].get("hit@10") == 0]
        still = [q for q in B if B[q].get("hit@10") == 0]
        print(f"\nBest dense config: {best}   (compared with best TF-IDF: {base})")
        print(f"  FIXED  (TF-IDF missed top-10, dense hits top-5): {len(fixed)}")
        for q in fixed: print(f"    {q} {B[q]['query'][:80]}")
        print(f"  BROKEN (TF-IDF hit top-5, dense misses top-10): {len(broke)}")
        for q in broke: print(f"    {q} {B[q]['query'][:80]}")
        print(f"  STILL MISSING top-10: {len(still)}")
        for q in still: print(f"    {q} [{B[q]['category']}] {B[q]['query'][:80]}")
        json.dump({k: all_rows[k] for k in (best, base)}, open(Path(a.results_dir) / "best_vs_tfidf_per_query.json", "w"), indent=1)
    print(f"\nSaved {Path(a.results_dir) / 'dense_summary.csv'}")

if __name__ == "__main__":
    main()