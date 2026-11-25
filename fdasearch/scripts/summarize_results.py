"""Turn results/metrics.csv + per-query files into a markdown report (results/REPORT.md)."""
import os, json, glob, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predx_dataset import load_dataset
RES = os.path.join(ROOT, "results")


def fmt(r, c):
    return f"{100*r[c]:.1f} [{100*r[c+'_lo']:.1f}, {100*r[c+'_hi']:.1f}]"


def main():
    m = pd.read_csv(os.path.join(RES, "metrics.csv"))
    m = m.drop_duplicates(subset=["method", "split"], keep="last")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None); ap.add_argument("--panel", default=None)
    a = ap.parse_args()
    corpus, queries, pairs, card = load_dataset(a.dataset, panel=a.panel)
    lines = ["# Predicate retrieval results: untuned baselines and fine-tuned EmbeddingGemma", ""]
    lines.append(f"Dataset: predx build of {card.get('built_at', '?')} (mask_level {card.get('mask_level')}, "
                 f"splits {card['config']['splits']['train'][0][:4]}-{card['config']['splits']['train'][1][:4]} / "
                 f"{card['config']['splits']['dev'][0][:4]} / {card['config']['splits']['test'][0][:4]}+). "
                 f"Corpus: {len(corpus)} 510(k) summaries ({corpus.date.min()} to {corpus.date.max()}). "
                 f"Queries: {len(queries)} retrievable subjects (>= 1 cited predicate in the corpus; "
                 f"dev: {(queries.split=='dev').sum()}, test: {(queries.split=='test').sum()}). "
                 f"Candidates for each query = corpus devices cleared before the subject's decision date; gold = all cited "
                 f"predicates in the corpus (primary + additional), `primary` columns score the primary predicate alone. "
                 f"Numbers are percentages with bootstrap 95% CIs over queries.")
    lines.append("")
    lines += ["Fine-tuned models (`emb:models/...`) are EmbeddingGemma-300M trained on the train pairs: "
              "`embeddinggemma-300m-ft` = 512 tokens (`checkpoint-646` epoch 1, `final` epoch 2); "
              "`-ft2` = the same run continued (checkpoints at 3, 4 and 5 total epochs); "
              "`-ft-eval1024` = the 512-token epoch 2 model searched with 1,024-token inputs; "
              "`-ft-len1024` = trained and searched at 1,024 tokens (`final` is the published model). "
              "`rrf:a|b` is reciprocal-rank fusion and `+pc` restricts candidates to the query's product code. "
              "The number of epochs and the checkpoint were chosen on dev.", ""]
    for split in ["test", "dev"]:
        g = m[m.split == split].sort_values("recall10", ascending=False)
        lines += [f"## {split} split, all in-corpus gold: primary + additional predicates (n={int(g.n.iloc[0])})", "",
                  "| method | Recall@10 | Hit@1 | Hit@10 | Hit@100 | MRR |", "|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            lines.append(f"| {r.method.replace(chr(124), chr(32) + chr(43) + chr(32))} | {fmt(r,'recall10')} | {fmt(r,'hit1')} | {fmt(r,'hit10')} | {fmt(r,'hit100')} | {fmt(r,'mrr')} |")
        lines.append("")
        g = g.sort_values("prim_hit10", ascending=False)
        npr = int(g.n_prim.iloc[0]) if "n_prim" in g else 0
        lines += [f"## {split} split, primary predicate only: the single predicate labelled primary (n={npr} queries whose primary is in the corpus)", "",
                  "Additional predicates and reference devices are ignored here: a hit means the primary itself was ranked in the top k.", "",
                  "| method | primary Hit@1 | primary Hit@10 | primary MRR |", "|---|---|---|---|"]
        for _, r in g.iterrows():
            lines.append(f"| {r.method.replace(chr(124), chr(32) + chr(43) + chr(32))} | {fmt(r,'prim_hit1')} | {fmt(r,'prim_hit10')} | {fmt(r,'prim_mrr')} |")
        lines.append("")
    # per-specialty breakdown for each method on test
    KEEP = ["tfidf", "bm25", "lsa", "pop", "ollama_embeddinggemma", "emb_BAAI__bge-m3", "emb_intfloat__e5-base-v2",
            "emb_Qwen__Qwen3-Embedding-4B", "gemini_gemini-embedding-001", "rrf_tfidf_ollama_embeddinggemma", "rrf_tfidf_gemini_gemini-embedding-001",
            "emb_google__embeddinggemma-300m", "emb_models__embeddinggemma-300m-ft-len1024__final",
            "rrf_tfidf_emb_models__embeddinggemma-300m-ft-len1024__final"]
    pq = {}
    for f in glob.glob(os.path.join(RES, "ranks", "*.per_query.csv")):
        name = os.path.basename(f).replace(".per_query.csv", "")
        if name in KEEP:
            pq[name] = pd.read_csv(f)
    if pq:
        qmeta = queries.set_index("k")
        lines += ["## Recall@10 by review panel (test split)", ""]
        tab = {}
        for name, d in pq.items():
            d = d[d.split == "test"].copy()
            d["panel"] = d.k.map(qmeta.panel)
            tab[name] = d.groupby("panel").recall10.mean()
        tab = pd.DataFrame(tab)
        cnt = queries[queries.split == "test"].groupby("panel").size()
        tab.insert(0, "n", cnt)
        tab = tab.sort_values("n", ascending=False)
        lines.append("| panel | n | " + " | ".join(tab.columns[1:]) + " |")
        lines.append("|---|---|" + "---|" * (len(tab.columns) - 1))
        for p, r in tab.iterrows():
            lines.append(f"| {p} | {int(r.n)} | " + " | ".join(f"{100*r[c]:.0f}" for c in tab.columns[1:]) + " |")
        lines.append("")
        # by candidate-pool size within product code
        lines += ["## Recall@10 by number of earlier same-product-code devices in corpus (test split)", ""]
        pc_count = dict(zip(queries.k, queries.n_same_code_candidates))
        bins = [0, 10, 50, 200, 10 ** 9]; labels = ["<10", "10-49", "50-199", "200+"]
        tab = {}
        for name, d in pq.items():
            d = d[d.split == "test"].copy()
            d["bin"] = pd.cut(d.k.map(pc_count), bins=bins, labels=labels, right=False)
            tab[name] = d.groupby("bin", observed=True).recall10.mean()
        tab = pd.DataFrame(tab)
        qt = queries[queries.split == "test"]
        tab.insert(0, "n", pd.cut(qt.k.map(pc_count), bins=bins, labels=labels, right=False).value_counts())
        lines.append("| same-code candidates | n | " + " | ".join(tab.columns[1:]) + " |")
        lines.append("|---|---|" + "---|" * (len(tab.columns) - 1))
        for p, r in tab.iterrows():
            lines.append(f"| {p} | {int(r.n)} | " + " | ".join(f"{100*r[c]:.0f}" for c in tab.columns[1:]) + " |")
        lines.append("")
    open(os.path.join(RES, "REPORT.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
