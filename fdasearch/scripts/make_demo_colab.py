"""Write notebooks/predicate_search_demo.ipynb: a self-contained Colab that downloads the published dataset and
fine-tuned model from Hugging Face, compares the model with TF-IDF and off-the-shelf embedders, and runs a search demo.

  python scripts/make_demo_colab.py
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ID = "Keyvanazami/embeddinggemma-300m-510k-predicate"
DATA_ID = "Keyvanazami/fda-510k-predicate-retrieval"

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)})


md(r'''
# 510(k) predicate search: fine-tuned EmbeddingGemma vs TF-IDF vs off-the-shelf embedders

Given a new medical device's **Indications for Use** and **Device Description**, rank earlier FDA 510(k) clearances so
that the predicate devices its summary cites come first.

This notebook:

1. downloads the benchmark ([dataset](https://huggingface.co/datasets/__DATA__)) and the fine-tuned model
   ([model](https://huggingface.co/__MODEL__)) from Hugging Face,
2. scores TF-IDF, five off-the-shelf embedders and the fine-tuned model on the held-out test split,
3. lets you search with your own device description.

**Runtime.** A GPU runtime is recommended (Runtime > Change runtime type > T4 GPU) but not required. With the default
settings everything finishes in about 5 minutes, because document vectors for every embedder are precomputed and
downloaded. No Hugging Face token or Google Drive access is needed.

Research demo. It recovers historical citations and is not regulatory advice.
''')

code(r'''
#@title Settings
MODEL_ID = "__MODEL__"
DATA_ID  = "__DATA__"
SPLIT    = "test"     #@param ["test", "dev"]
# Off-the-shelf models to re-embed on this runtime instead of using the downloaded vectors (slow: 52,123 documents each).
# Example: RECOMPUTE = ["sentence-transformers/all-MiniLM-L6-v2"]. google/embeddinggemma-300m is gated and needs an HF_TOKEN secret.
RECOMPUTE = []
''')

code(r'''
#@title Install and import
%pip install -q -U "sentence-transformers>=5.1" huggingface_hub scikit-learn
import gzip, json, os, re, time
import numpy as np, pandas as pd, torch
from huggingface_hub import snapshot_download
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")
''')

md(r'''
## 1. Data

`corpus.jsonl.gz` has one earlier device per row, sorted by clearance date. Each query is a later device with the
predicates its 510(k) summary cites (`gold_in_corpus`) and the number of corpus rows cleared before it
(`n_candidates`). A method may only rank those earlier rows.
''')

code(r'''
#@title Download and load the dataset
D = snapshot_download(DATA_ID, repo_type="dataset")

def read_jsonl(name):
    with gzip.open(os.path.join(D, name), "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def make_text(ifu, description):
    parts = []
    if ifu: parts.append("Indications for Use: " + ifu)
    if description: parts.append("Device Description: " + description)
    return "\n".join(parts).strip()

corpus = pd.DataFrame(read_jsonl("corpus.jsonl.gz"))
corpus["text"] = [make_text(a, b) for a, b in zip(corpus.ifu, corpus.description)]
kidx = {k: i for i, k in enumerate(corpus.k)}

order = json.load(open(os.path.join(D, "embeddings", "index.json")))["queries_order"]   # row order of the *.queries.npy files
qpos = {k: i for i, k in enumerate(order)}
queries = pd.DataFrame([q for q in read_jsonl(f"queries_{SPLIT}.jsonl.gz") if q["gold_in_corpus"]])
queries["text"] = [make_text(a, b) for a, b in zip(queries.ifu, queries.description)]
queries["primary"] = [p[0] if p else None for p in queries.gold_primary_in_corpus]
print(f"corpus: {len(corpus):,} devices, {corpus.date.min()} to {corpus.date.max()}")
print(f"{SPLIT} queries: {len(queries):,} devices cleared {queries.date.min()} to {queries.date.max()}, "
      f"{queries.gold_in_corpus.map(len).mean():.2f} cited predicates each on average")
queries[["k", "date", "product_code", "gold_in_corpus", "n_candidates"]].head()
''')

md(r'''
## 2. Scoring

Search is exact: every query is compared with every candidate. For each query we record the rank of each cited
predicate among the devices cleared before it, then report

- **Recall@10**: share of a query's cited predicates in the top 10,
- **Hit@k**: at least one cited predicate in the top k,
- **primary Hit@k**: the single primary predicate in the top k.
''')

code(r'''
#@title Evaluation functions
N = len(corpus)
GOLD = [[kidx[p] for p in g] for g in queries.gold_in_corpus]
PRIM = [kidx[p] if isinstance(p, str) else -1 for p in queries.primary]
NCAND = queries.n_candidates.to_numpy()
SELF = [kidx.get(k, -1) for k in queries.k]

def rank_all(score_rows, chunk=256):
    """score_rows(lo, hi) -> float tensor [hi-lo, N] of scores. Returns per-query gold ranks, primary rank, top-100 ids."""
    idx = torch.arange(N, device=DEVICE)
    ranks, prim, top = [], [], []
    for lo in range(0, len(queries), chunk):
        hi = min(lo + chunk, len(queries))
        s = score_rows(lo, hi).to(DEVICE).float()
        s[idx[None, :] >= torch.as_tensor(NCAND[lo:hi], device=DEVICE)[:, None]] = -float("inf")   # only earlier devices
        for r, j in enumerate(SELF[lo:hi]):
            if j >= 0: s[r, j] = -float("inf")
        top.append(torch.topk(s, 100, dim=1).indices.cpu().numpy())
        for r in range(hi - lo):
            def rk(g):   # 1 + number of candidates scored higher (ties broken by corpus order)
                v = s[r, g]
                if not torch.isfinite(v): return 10 ** 9
                return int(1 + (s[r] > v).sum() + ((s[r] == v) & (idx < g)).sum())
            ranks.append(sorted(rk(g) for g in GOLD[lo + r]))
            prim.append(rk(PRIM[lo + r]) if PRIM[lo + r] >= 0 else None)
    return ranks, prim, np.concatenate(top)

def metrics(ranks, prim):
    p = [x for x in prim if x is not None]
    return {"Recall@10": np.mean([np.mean([r <= 10 for r in rs]) for rs in ranks]),
            "Hit@1": np.mean([rs[0] <= 1 for rs in ranks]), "Hit@10": np.mean([rs[0] <= 10 for rs in ranks]),
            "Hit@100": np.mean([rs[0] <= 100 for rs in ranks]),
            "primary Hit@1": np.mean([x <= 1 for x in p]), "primary Hit@10": np.mean([x <= 10 for x in p])}

RESULTS, TOP = {}, {}
def run(name, score_rows):
    t0 = time.time()
    ranks, prim, top = rank_all(score_rows)
    RESULTS[name], TOP[name] = metrics(ranks, prim), top
    print(f"{name:45s} Recall@10 {100 * RESULTS[name]['Recall@10']:.1f}   Hit@1 {100 * RESULTS[name]['Hit@1']:.1f}   ({time.time() - t0:.0f}s)")
''')

md(r'''
## 3. TF-IDF baseline

Word and two-word-phrase TF-IDF with cosine similarity, fitted on the corpus. It is the strongest method that needs no training.
''')

code(r'''
#@title TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.5, stop_words="english", ngram_range=(1, 2))
D_tfidf = tv.fit_transform(corpus.text.tolist())
DT_tfidf = D_tfidf.T.tocsr()
Q_tfidf = tv.transform(queries.text.tolist())
run("TF-IDF", lambda lo, hi: torch.from_numpy((Q_tfidf[lo:hi] @ DT_tfidf).toarray()))
''')

md(r'''
## 4. Off-the-shelf embedders

Untuned embedding models, each with its recommended query prompt and 512-token inputs. By default their vectors are
downloaded; list a model in `RECOMPUTE` (Settings) to embed the corpus on this runtime instead.
''')

code(r'''
#@title Off-the-shelf embedders
from sentence_transformers import SentenceTransformer
BASELINES = {   # model id: (query prefix, document prefix)
    "sentence-transformers/all-MiniLM-L6-v2": ("", ""),
    "BAAI/bge-base-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "BAAI/bge-m3": ("", ""),
    "Qwen/Qwen3-Embedding-0.6B": ("Instruct: Given a medical device description, retrieve FDA 510(k) predicate devices that are substantially equivalent to it\nQuery: ", ""),
    "google/embeddinggemma-300m": ("task: search result | query: ", "title: none | text: "),
}
QROWS = [qpos[k] for k in queries.k]

def load_vectors(tag):
    d = np.load(os.path.join(D, "embeddings", f"{tag}.docs.npy")).astype(np.float32)
    q = np.load(os.path.join(D, "embeddings", f"{tag}.queries.npy")).astype(np.float32)[QROWS]
    norm = lambda x: torch.nn.functional.normalize(torch.from_numpy(x).to(DEVICE), dim=1)   # stored as float16
    return norm(d), norm(q)

def embed(model_id, texts, prefix, max_len=512, batch=32):
    m = SentenceTransformer(model_id, device=DEVICE, trust_remote_code=True)
    m.max_seq_length = max_len
    v = m.encode([prefix + t for t in texts], batch_size=batch, normalize_embeddings=True, show_progress_bar=True, convert_to_tensor=True)
    del m; torch.cuda.empty_cache()
    return v.float()

for model_id, (qp, dp) in BASELINES.items():
    if model_id in RECOMPUTE:
        Dv, Qv = embed(model_id, corpus.text.tolist(), dp), embed(model_id, queries.text.tolist(), qp)
    else:
        Dv, Qv = load_vectors(model_id.replace("/", "__"))
    run(model_id + (" (recomputed)" if model_id in RECOMPUTE else ""), lambda lo, hi: Qv[lo:hi] @ Dv.T)
''')

md(r'''
## 5. Fine-tuned model

EmbeddingGemma-300M after two epochs of contrastive training on 41,281 (device, cited predicate) pairs from 2015 to
2023, with hard negatives mined by TF-IDF and 1,024-token inputs. The model is downloaded and used live for the
queries of the demo below. For the benchmark, its corpus vectors are downloaded; the cell first re-embeds a sample
of documents and queries to confirm that the downloaded vectors match what this model produces here.
''')

code(r'''
#@title Load the fine-tuned model and score it
ft = SentenceTransformer(MODEL_ID, device=DEVICE)
ft.max_seq_length = 1024
D_ft, Q_ft = load_vectors("finetuned-embeddinggemma-300m-510k")

sample = np.random.default_rng(0).choice(N, 64, replace=False)
live = ft.encode(corpus.text.iloc[sample].tolist(), prompt_name="document", batch_size=8, normalize_embeddings=True, convert_to_tensor=True).float()
print(f"downloaded vs live document vectors, cosine: mean {(live * D_ft[sample]).sum(1).mean():.4f}, min {(live * D_ft[sample]).sum(1).min():.4f}")
live_q = ft.encode(queries.text.iloc[:32].tolist(), prompt_name="query", batch_size=8, normalize_embeddings=True, convert_to_tensor=True).float()
print(f"downloaded vs live query vectors, cosine:    mean {(live_q * Q_ft[:32]).sum(1).mean():.4f}")

run("Fine-tuned EmbeddingGemma", lambda lo, hi: Q_ft[lo:hi] @ D_ft.T)
''')

md(r'''
## 6. Fusion and comparison

Reciprocal-rank fusion combines two ranked lists: each device scores `1 / (60 + rank)` in each list's top 100, and the
sums are ranked. Lexical and embedding search make different mistakes, so the fused list is better than either.
''')

code(r'''
#@title Fuse with TF-IDF and compare
def rrf(a, b, k=60):
    w = torch.zeros(len(queries), N)
    for t in (TOP[a], TOP[b]):
        w.scatter_add_(1, torch.from_numpy(t).long(), (1.0 / (k + torch.arange(1, 101))).repeat(len(queries), 1))
    return lambda lo, hi: w[lo:hi].clone()

run("Fine-tuned EmbeddingGemma + TF-IDF (fusion)", rrf("Fine-tuned EmbeddingGemma", "TF-IDF"))
run("google/embeddinggemma-300m + TF-IDF (fusion)", rrf([n for n in RESULTS if n.startswith("google/embeddinggemma-300m")][0], "TF-IDF"))

table = (pd.DataFrame(RESULTS).T * 100).round(1).sort_values("Recall@10", ascending=False)
table
''')

code(r'''
#@title Chart
import matplotlib.pyplot as plt
t = table.sort_values("Recall@10")
colors = ["#1f5fbf" if "Fine-tuned" in n else "#8a94a6" if n != "TF-IDF" else "#d9822b" for n in t.index]
fig, ax = plt.subplots(figsize=(9, 0.45 * len(t) + 1))
bars = ax.barh(t.index, t["Recall@10"], color=colors)
ax.bar_label(bars, fmt="%.1f", padding=3)
ax.set_xlabel(f"Recall@10 on the {SPLIT} split (%)"); ax.set_xlim(0, 100)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.show()
''')

md(r'''
## 7. Search demo

Describe a device and run the cell. Results are earlier clearances ranked by the fine-tuned model fused with TF-IDF,
each with a link to its FDA record. Leave both boxes empty to load a random held-out test device; its cited predicates
are then marked in the results.
''')

code(r'''
#@title Search
from IPython.display import HTML, display
indications = ""  #@param {type:"string"}
description = ""  #@param {type:"string"}
method = "fine-tuned + TF-IDF"  #@param ["fine-tuned + TF-IDF", "fine-tuned", "TF-IDF"]
product_code = ""  #@param {type:"string"}
results = 10  #@param {type:"slider", min:5, max:50, step:5}

KNUM = re.compile(r"(?<![A-Za-z0-9])(?:K|BK|DEN|P)\d{6}(?![0-9])")
DATES, PCS = corpus.date.to_numpy(), corpus.product_code.to_numpy()

def search(ifu, desc, method="fine-tuned + TF-IDF", before=None, pc="", topk=10):
    text = KNUM.sub("[K-NUMBER]", make_text(ifu, desc))        # the corpus has K-numbers masked the same way
    mask = np.ones(N, dtype=bool)
    if before: mask &= DATES < before
    if pc: mask &= PCS == pc.strip().upper()
    lists = []
    if "fine-tuned" in method:
        q = ft.encode([text], prompt_name="query", normalize_embeddings=True, convert_to_tensor=True).float()
        lists.append((q @ D_ft.T)[0].cpu().numpy())
    if "TF-IDF" in method:
        lists.append((tv.transform([text]) @ DT_tfidf).toarray().ravel())
    if len(lists) == 1:
        s = np.where(mask, lists[0], -np.inf)
    else:
        s = np.zeros(N)
        for x in lists:
            top = np.argsort(-np.where(mask, x, -np.inf), kind="stable")[:100]
            s[top] += 1.0 / (60 + np.arange(1, len(top) + 1))
        s = np.where(mask & (s > 0), s, -np.inf)
    out = np.argsort(-s, kind="stable")
    return out[np.isfinite(s[out])]

gold, before = [], None
if not indications.strip() and not description.strip():
    ex = queries.sample(1).iloc[0]
    indications, description, gold, before = ex.ifu, ex.description, list(ex.gold_in_corpus), ex.date
    print(f"Random test device {ex.k}, cleared {ex.date}. Its summary cites: {', '.join(gold)}")
    print("\nIndications for Use:", indications[:500], "\n\nDevice Description:", description[:700], "\n")

hits = search(indications, description, method, before, product_code, results)
rank = {int(i): n + 1 for n, i in enumerate(hits)}
if gold:
    print("Ranks of the cited predicates:", ", ".join(f"{g}: {rank.get(kidx[g], 'not in the ranked list')}" for g in gold))
esc = lambda x: str(x).replace("&", "&amp;").replace("<", "&lt;")
rows = "".join(
    f"<tr style='background:{'#e6f4ea' if r.k in gold else 'transparent'};color:{'#111' if r.k in gold else 'inherit'}'><td>{n + 1}</td>"
    f"<td><a target='_blank' href='https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID={r.k}'>{r.k}</a>{' (cited)' if r.k in gold else ''}</td>"
    f"<td>{esc(r.device_name)}</td><td>{esc(r.applicant)}</td><td>{r.date}</td><td>{r.product_code}</td><td>{esc(r.ifu[:260])}</td></tr>"
    for n, r in enumerate(corpus.iloc[hits[:results]].itertuples()))
display(HTML("<table style='font-size:13px;text-align:left'><tr><th>#</th><th>510(k)</th><th>Device</th><th>Applicant</th><th>Cleared</th><th>Code</th><th>Indications for Use</th></tr>" + rows + "</table>"))
''')

md(r'''
### Optional: a shareable web page

The next cell starts a small Gradio app with the same search and prints a temporary public link (valid while this
runtime is running). Skip it if you only need the cell above.
''')

code(r'''
#@title Launch a Gradio app (optional)
%pip install -q gradio
import gradio as gr

def ui_search(ifu, desc, method, pc, topk):
    if not (ifu or "").strip() and not (desc or "").strip():
        return pd.DataFrame()
    hits = search(ifu, desc, method, None, pc, int(topk))[: int(topk)]
    r = corpus.iloc[hits]
    return pd.DataFrame({"510(k)": r.k, "Device": r.device_name, "Applicant": r.applicant, "Cleared": r.date,
                         "Code": r.product_code, "Indications for Use": r.ifu.str[:300]})

demo = gr.Interface(
    ui_search,
    [gr.Textbox(label="Indications for Use", lines=4), gr.Textbox(label="Device Description", lines=6),
     gr.Radio(["fine-tuned + TF-IDF", "fine-tuned", "TF-IDF"], value="fine-tuned + TF-IDF", label="Method"),
     gr.Textbox(label="Product code (optional)"), gr.Slider(5, 50, value=10, step=5, label="Results")],
    gr.Dataframe(label="Candidate predicates", wrap=True),
    title="510(k) predicate search", description="Research demo. Not regulatory advice.", flagging_mode="never")
demo.launch(share=True)
''')

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}}
for c in nb["cells"]:
    c["source"] = [s.replace("__MODEL__", MODEL_ID).replace("__DATA__", DATA_ID) for s in c["source"]]
out = os.path.join(ROOT, "notebooks", "predicate_search_demo.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote", out, len(cells), "cells")
