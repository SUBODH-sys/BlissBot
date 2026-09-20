import importlib.util
from pathlib import Path

harness_path = Path("/workspaces/BlissBot/03_eval_harness.py")
spec = importlib.util.spec_from_file_location("eval_harness", harness_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load evaluation harness from {harness_path}")
_eval_harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_eval_harness)
load = _eval_harness.load
evaluate = _eval_harness.evaluate

corpus_path = "/workspaces/BlissBot/out_dir/bliss_corpus_clean.json"
golden_path = "/workspaces/BlissBot/out_dir/golden_eval_set.json"


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
        for r in miss:
            print(f"  {r['qid']} [{r['category']}] {r['query']}")
