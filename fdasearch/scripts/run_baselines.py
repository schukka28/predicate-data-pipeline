"""Zero-shot predicate retrieval baselines on a predx build (see predx_dataset.py for the dataset contract).

Usage: python run_baselines.py --methods bm25,lsa,pop,emb:BAAI/bge-base-en-v1.5,ollama:embeddinggemma [--pc] [--dataset DIR]
Writes results/metrics.csv (appends) and results/ranks/{method}.json (rank of each positive per query).
Candidates of a query = corpus[:n_candidates] (devices dated strictly before the subject), gold = gold_in_corpus
(primary + additional predicates present in the corpus); `prim_*` columns score the primary predicate alone.
"""
import argparse, json, os, re, sys, time, math
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predx_dataset import load_dataset
RES = os.path.join(ROOT, "results")
os.makedirs(os.path.join(RES, "ranks"), exist_ok=True)
os.makedirs(os.path.join(RES, "emb"), exist_ok=True)

# model-specific prompts (query, document)
PROMPTS = {
    "intfloat/e5-base-v2": ("query: ", "passage: "),
    "intfloat/e5-large-v2": ("query: ", "passage: "),
    "BAAI/bge-base-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "BAAI/bge-large-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "BAAI/bge-m3": ("", ""),
    "nomic-ai/nomic-embed-text-v1.5": ("search_query: ", "search_document: "),
    "Qwen/Qwen3-Embedding-0.6B": ("Instruct: Given a medical device description, retrieve FDA 510(k) predicate devices that are substantially equivalent to it\nQuery: ", ""),
    "Qwen/Qwen3-Embedding-4B": ("Instruct: Given a medical device description, retrieve FDA 510(k) predicate devices that are substantially equivalent to it\nQuery: ", ""),
    "Alibaba-NLP/gte-modernbert-base": ("", ""),
    "sentence-transformers/all-MiniLM-L6-v2": ("", ""),
    "allenai/specter2_base": ("", ""),
    "Snowflake/snowflake-arctic-embed-m-v2.0": ("query: ", ""),
    "google/embeddinggemma-300m": ("task: search result | query: ", "title: none | text: "),
}
def prompts_for(model_name):
    """(query prefix, document prefix). A local fine-tuned checkpoint keeps its base model's prompts: the base is
    matched by its short name appearing in the path (models/eg300m-ft/final -> embeddinggemma via train_config.json)."""
    if model_name in PROMPTS:
        return PROMPTS[model_name]
    cfg = os.path.join(os.path.dirname(os.path.normpath(model_name)), "train_config.json")
    for c in (cfg, os.path.join(model_name, "train_config.json")):
        if os.path.exists(c):
            base = json.load(open(c)).get("model") or ""
            return prompts_for(base) if base != model_name else ("", "")
    return ("", "")


OLLAMA_PROMPTS = {"embeddinggemma": ("task: search result | query: ", "title: none | text: "),
                  "nomic-embed-text": ("search_query: ", "search_document: ")}


def embed_st(model_name, texts, is_query, batch=64, max_len=512):
    from sentence_transformers import SentenceTransformer
    import torch
    kw = {"trust_remote_code": True}
    if "Qwen3" in model_name or "gemma" in model_name:
        # bf16 where the GPU supports it (Ampere+); older GPUs such as Colab's T4 fall back to fp32
        kw["model_kwargs"] = {"torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32}
    m = SentenceTransformer(model_name, device="cuda", **kw)
    m.max_seq_length = max_len
    qp, dp = prompts_for(model_name)
    pre = qp if is_query else dp
    out = m.encode([pre + t for t in texts], batch_size=batch, normalize_embeddings=True,
                   show_progress_bar=True, convert_to_numpy=True)
    del m; torch.cuda.empty_cache()
    return out.astype(np.float32)


def _ollama_post(model, chunk, tries=5):
    import requests
    for a in range(tries):
        try:
            r = requests.post("http://localhost:11434/api/embed", json={"model": model, "input": chunk, "truncate": True}, timeout=600)
            if r.status_code == 200:
                return r.json()["embeddings"]
            err = f"{r.status_code} {r.text[:120]}"
        except Exception as e:
            err = str(e)[:120]
        time.sleep(3 * (a + 1))
    if len(chunk) > 1:  # split and retry halves
        h = len(chunk) // 2
        return _ollama_post(model, chunk[:h]) + _ollama_post(model, chunk[h:])
    if len(chunk[0]) > 1500:  # single bad item: shorten
        return _ollama_post(model, [chunk[0][:1500]])
    print("  ollama giving up on one item:", err, flush=True)
    return [None]


def embed_ollama(model, texts, is_query, batch=32):
    qp, dp = OLLAMA_PROMPTS.get(model.split(":")[0], ("", ""))
    pre = qp if is_query else dp
    out = []
    for i in range(0, len(texts), batch):
        chunk = [pre + t[:6000] for t in texts[i:i + batch]]
        out.extend(_ollama_post(model, chunk))
        if (i // batch) % 50 == 0:
            print(f"  ollama {i}/{len(texts)}", flush=True)
    dim = len(next(v for v in out if v is not None))
    v = np.asarray([x if x is not None else [0.0] * dim for x in out], dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def embed_gemini(model, texts, is_query, batch=100, dim=3072):
    """Gemini embeddings via google-genai. Key from GEMINI_API_KEY or secrets/gemini_key.txt."""
    from google import genai
    from google.genai import types
    envf = os.path.join(ROOT, ".env")
    if os.path.exists(envf):
        for line in open(envf):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                if v and not os.environ.get(k.strip()):
                    os.environ[k.strip()] = v
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("No Gemini key: put GEMINI_API_KEY=... in E:\fdasearch\.env")
    client = genai.Client(api_key=key)
    cfg = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT", output_dimensionality=dim)
    out = []
    for i in range(0, len(texts), batch):
        chunk = [t[:8000] for t in texts[i:i + batch]]
        for attempt in range(8):
            try:
                r = client.models.embed_content(model=model, contents=chunk, config=cfg)
                out.extend([e.values for e in r.embeddings]); break
            except Exception as e:
                msg = str(e)[:160]
                if attempt == 7:
                    raise
                time.sleep(min(60, 5 * 2 ** attempt))
                print(f"  gemini retry {attempt+1} at {i}: {msg}", flush=True)
        if (i // batch) % 20 == 0:
            print(f"  gemini {i}/{len(texts)}", flush=True)
    v = np.asarray(out, dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def cached_embed(tag, fn, texts, keys, is_query, chunk=4000):
    """Embed only texts whose key is not yet cached; cache keyed by K-number (npz), saved every `chunk` texts."""
    p = os.path.join(RES, "emb", f"{tag}.{'q' if is_query else 'd'}.npz")
    have = {}
    if os.path.exists(p):
        with np.load(p, allow_pickle=False) as z:  # close the file: Windows cannot replace an open one
            have = dict(zip(z["keys"].tolist(), z["vecs"]))
    todo = [(k, t) for k, t in zip(keys, texts) if k not in have]
    if todo:
        print(f"  embedding {len(todo)} new of {len(texts)} ({len(have)} cached)", flush=True)
        for i in range(0, len(todo), chunk):
            part = todo[i:i + chunk]
            v = fn([t for _, t in part], is_query)
            for (k, _), vec in zip(part, v):
                have[k] = vec
            tmp = p[:-4] + ".tmp.npz"  # write then rename, so an interrupted run cannot leave a truncated cache
            np.savez(tmp, keys=np.array(list(have.keys())), vecs=np.stack(list(have.values())))
            os.replace(tmp, p)
    return np.stack([have[k] for k in keys]).astype(np.float32)


def evaluate(scores_fn, corpus, queries, method, pc_filter=False):
    """scores_fn(qi) -> np.array of scores over corpus rows (higher better)."""
    kidx = {k: i for i, k in enumerate(corpus.k)}
    pcs = corpus.product_code.values
    pos_idx = np.arange(len(corpus))
    rows = []
    rank_dump = {}
    top = np.full((len(queries), 100), -1, dtype=np.int32)
    for qi, q in enumerate(queries.itertuples()):
        s = np.asarray(scores_fn(qi), dtype=np.float64).copy()
        valid = pos_idx < q.n_candidates  # predx: corpus sorted by date, candidates = rows dated before the subject
        if pc_filter:
            valid &= pcs == q.product_code
        s[~valid] = -np.inf
        if q.k in kidx:
            s[kidx[q.k]] = -np.inf
        order = np.argsort(-s, kind="stable")
        top[qi] = order[:100]
        rank_of = np.empty(len(s), dtype=np.int64); rank_of[order] = np.arange(1, len(s) + 1)
        pos = [kidx[p] for p in q.positives]
        pr = sorted(int(rank_of[i]) for i in pos)
        n_valid = int(valid.sum())
        pr = [r if r <= n_valid else 10 ** 9 for r in pr]  # excluded by filter -> not retrievable
        prim = int(rank_of[kidx[q.primary]]) if isinstance(q.primary, str) else None
        if prim is not None and prim > n_valid: prim = 10 ** 9
        rank_dump[q.k] = {"pos": pr, "primary": prim, "n_valid": n_valid}
        rows.append(dict(
            k=q.k, split=q.split, n_pos=len(pr),
            recall10=sum(r <= 10 for r in pr) / len(pr), recall50=sum(r <= 50 for r in pr) / len(pr),
            hit1=float(pr[0] <= 1), hit10=float(pr[0] <= 10), hit100=float(pr[0] <= 100), mrr=1.0 / pr[0],
            prim_hit1=(None if prim is None else float(prim <= 1)), prim_hit10=(None if prim is None else float(prim <= 10)),
            prim_mrr=(None if prim is None else 1.0 / prim),
        ))
    df = pd.DataFrame(rows)
    json.dump(rank_dump, open(os.path.join(RES, "ranks", f"{method}.json"), "w"))
    os.makedirs(os.path.join(RES, "top"), exist_ok=True)
    np.save(os.path.join(RES, "top", f"{method}.npy"), top)
    df.to_csv(os.path.join(RES, "ranks", f"{method}.per_query.csv"), index=False)
    return df


def rrf_scores_fn(names, n_docs, k=60):
    """Reciprocal rank fusion over saved top-100 lists of previously run methods."""
    tops = [np.load(os.path.join(RES, "top", f"{n}.npy")) for n in names]
    def fn(qi):
        s = np.zeros(n_docs)
        for t in tops:
            idx = t[qi]; idx = idx[idx >= 0]
            s[idx] += 1.0 / (k + np.arange(1, len(idx) + 1))
        return s
    return fn


def bootstrap_ci(x, n=1000, seed=0):
    x = np.asarray(x, dtype=float); rng = np.random.default_rng(seed)
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(n)])
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def summarize(df, method, note=""):
    out = []
    for split, g in df.groupby("split"):
        rec = dict(method=method, split=split, n=len(g), n_prim=int(g["prim_hit1"].notna().sum()), note=note)
        for c in ["recall10", "recall50", "hit1", "hit10", "hit100", "mrr", "prim_hit1", "prim_hit10", "prim_mrr"]:
            v = g[c].dropna()
            lo, hi = bootstrap_ci(v)
            rec[c] = round(v.mean(), 4); rec[c + "_lo"] = round(lo, 4); rec[c + "_hi"] = round(hi, 4)
        out.append(rec)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", required=True)
    ap.add_argument("--pc", action="store_true", help="also run each method restricted to subject's product code")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dataset", default=None, help="predx build directory (default: $PREDX_DATASET or data/dataset)")
    ap.add_argument("--panel", default=None, help="restrict corpus and queries to one review panel code (quick runs)")
    a = ap.parse_args()
    corpus, queries, pairs, card = load_dataset(a.dataset, panel=a.panel)
    if a.limit:
        queries = queries.sample(a.limit, random_state=0).reset_index(drop=True)
    print(f"corpus {len(corpus)} docs, {len(queries)} queries", flush=True)
    doc_texts = corpus.text.tolist(); q_texts = queries["query"].tolist()

    for spec in a.methods.split(","):
        t0 = time.time()
        name = spec
        if spec == "bm25":
            import bm25s, Stemmer
            st = Stemmer.Stemmer("english")
            ct = bm25s.tokenize(doc_texts, stopwords="en", stemmer=st)
            idx = bm25s.BM25(); idx.index(ct)
            qt = bm25s.tokenize(q_texts, stopwords="en", stemmer=st, return_ids=False)
            def fn(qi, qt=qt): return idx.get_scores(qt[qi])
        elif spec == "lsa":
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            from sklearn.preprocessing import normalize
            tv = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.5, stop_words="english")
            D = tv.fit_transform(doc_texts); Q = tv.transform(q_texts)
            svd = TruncatedSVD(300, random_state=0); Dl = normalize(svd.fit_transform(D)); Ql = normalize(svd.transform(Q))
            def fn(qi): return Ql[qi] @ Dl.T
        elif spec == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            tv = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.5, stop_words="english", ngram_range=(1, 2))
            D = tv.fit_transform(doc_texts); Q = tv.transform(q_texts)
            DT = D.T.tocsr()  # convert once: Q[qi] @ D.T re-converted the whole matrix on every query
            def fn(qi, Q=Q, DT=DT): return (Q[qi] @ DT).toarray().ravel()
        elif spec == "pop":
            # citation popularity within product code (predx product_code_popularity): counts from the train pairs
            cnt = {}
            for pr in pairs:
                cnt[pr["p"]] = cnt.get(pr["p"], 0) + 1
            base = np.array([cnt.get(k, 0) for k in corpus.k], dtype=float)
            recency = np.argsort(np.argsort(corpus.date.values)) / len(corpus)  # tie-break by recency
            pcs = corpus.product_code.values
            def fn(qi): return np.where(pcs == queries.product_code.iloc[qi], base + recency, -1.0)
        elif spec == "random":
            rng = np.random.default_rng(0)
            def fn(qi): return rng.random(len(corpus))
        elif spec.startswith(("emb:", "ollama:", "gemini:")):
            kind, model = spec.split(":", 1)
            tag = model.replace("/", "__")
            if kind == "emb":
                f = lambda t, isq: embed_st(model, t, isq, max_len=a.max_len)
            elif kind == "gemini":
                f = lambda t, isq: embed_gemini(model, t, isq)
            else:
                f = lambda t, isq: embed_ollama(model, t, isq)
            D = cached_embed(tag, f, doc_texts, corpus.k.tolist(), False)
            Q = cached_embed(tag, f, q_texts, queries.k.tolist(), True)
            def fn(qi): return Q[qi] @ D.T
            name = spec
        elif spec.startswith("rrf:"):
            # e.g. rrf:bm25|emb_BAAI__bge-base-en-v1.5  (names as saved under results/top)
            fn = rrf_scores_fn(spec[4:].split("|"), len(corpus))
        else:
            print("unknown", spec); continue
        for pc in ([False, True] if a.pc else [False]):
            mname = name + ("+pc" if pc else "")
            df = evaluate(fn, corpus, queries, re.sub(r"[^A-Za-z0-9_.+-]", "_", mname.replace("/", "__")), pc_filter=pc)
            s = summarize(df, mname)
            s["seconds"] = round(time.time() - t0, 1)
            print(s[["method", "split", "n", "recall10", "hit1", "hit10", "mrr", "prim_hit10"]].to_string(index=False), flush=True)
            p = os.path.join(RES, "metrics.csv")
            s.to_csv(p, mode="a", header=not os.path.exists(p), index=False)


if __name__ == "__main__":
    main()
