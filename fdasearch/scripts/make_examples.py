import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from predx_dataset import load_dataset
corpus, queries, _, card = load_dataset("E:/predx_data/dataset")
kidx = {k: i for i, k in enumerate(corpus.k)}
M = {"Fine-tuned + TF-IDF": "rrf_tfidf_emb_models__embeddinggemma-300m-ft-len1024__final",
     "Fine-tuned EmbeddingGemma": "emb_models__embeddinggemma-300m-ft-len1024__final",
     "TF-IDF": "tfidf",
     "EmbeddingGemma, untuned": "emb_google__embeddinggemma-300m"}
top = {n: np.load(f"results/top/{t}.npy") for n, t in M.items()}
ranks = {n: json.load(open(f"results/ranks/{t}.json")) for n, t in M.items()}
test = queries[queries.split == "test"]
FDA = "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID="

# full per-query table
rows = []
for qi, q in test.iterrows():
    r = {"k": q.k, "device_name": corpus.device_name[kidx[q.k]] if q.k in kidx else "", "date": q.date, "panel": q.panel,
         "product_code": q.product_code, "n_candidates": q.n_candidates, "cited_in_corpus": " ".join(q.positives), "primary": q.primary or ""}
    for n in M:
        pr = ranks[n][q.k]["pos"]
        r[f"{n} | best rank"] = pr[0] if pr[0] < 10 ** 9 else ""
        r[f"{n} | recall@10"] = round(sum(x <= 10 for x in pr) / len(pr), 3)
        r[f"{n} | top 10"] = " ".join(corpus.k[i] for i in top[n][qi][:10] if i >= 0)
    rows.append(r)
pq = pd.DataFrame(rows, index=test.index)
pq.to_csv("results/eval_test_per_query.csv", index=False, encoding="utf-8-sig")
print("per-query rows", len(pq))

F, T = "Fine-tuned + TF-IDF", "TF-IDF"
br = lambda n: pd.to_numeric(pq[f"{n} | best rank"], errors="coerce").fillna(10 ** 9)
buckets = [("Fine-tuned search finds a cited predicate in the top 10; TF-IDF does not", (br(F) <= 10) & (br(T) > 10), 4),
           ("Both find a cited predicate in the top 10", (br(F) <= 10) & (br(T) <= 10), 3),
           ("TF-IDF finds a cited predicate in the top 10; fine-tuned search does not", (br(F) > 10) & (br(T) <= 10), 2),
           ("Neither finds a cited predicate in the top 10", (br(F) > 10) & (br(T) > 10), 3)]
clip = lambda s, n: (s[:n].rsplit(" ", 1)[0] + " ...") if len(s) > n else s
cell = lambda s: str(s).replace("|", "/").replace("\n", " ")
L = ["# Example queries and results", "",
     f"Twelve devices from the test split (cleared 2025 to 2026), with the top 5 results of four methods. Dataset: predx build of {card['built_at'][:10]}, {len(corpus):,} corpus devices.",
     "",
     "The examples are a random draw (seed 0) inside four groups, so they illustrate behaviour and are not a representative sample. Share of all 5,308 test queries in each group:", ""]
for title, m, n in buckets:
    L.append(f"- {title}: {100 * m.mean():.1f}%")
L += ["", "A result marked **CITED** is a predicate named in the device's own 510(k) summary. Query text is shown as the models see it: K-numbers are masked as `[K-NUMBER]` and names of cited predicates as `[DEVICE]`. Long text is cut here for readability.", ""]
ex = 0
for title, m, n in buckets:
    L += [f"## {title}", ""]
    for qi in pq[m].sample(n, random_state=0).index:
        ex += 1
        q = test.loc[qi]; gold = set(q.positives)
        L += [f"### Example {ex}: [{q.k}]({FDA}{q.k}) {cell(pq.device_name[qi])}", "",
              f"Cleared {q.date} · product code {q.product_code} · panel {q.panel} · {q.n_candidates:,} earlier candidates", "",
              "> " + clip(q.query, 900).replace("\n", "\n>\n> "), "",
              "**Cited predicates in the corpus:** " + "; ".join(
                  f"[{g}]({FDA}{g}) {cell(corpus.device_name[kidx[g]])}" + (" (primary)" if g == q.primary else "") for g in q.positives), "",
              "| Method | Rank of best cited predicate | Top 5 |", "| --- | --- | --- |"]
        for name in M:
            b = pq[f"{name} | best rank"][qi]
            items = []
            for i in top[name][qi][:5]:
                k = corpus.k[i]
                s = f"{k} {cell(clip(corpus.device_name[i], 45))}"
                items.append(f"**{s} CITED**" if k in gold else s)
            L.append(f"| {name} | {b if b != '' else 'not ranked'} | " + "<br>".join(f"{j + 1}. {s}" for j, s in enumerate(items)) + " |")
        L.append("")
open("results/EXAMPLES.md", "w", encoding="utf-8").write("\n".join(L))
print("examples", ex)
