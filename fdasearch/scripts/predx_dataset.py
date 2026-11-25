"""Read a predx build (data/dataset/ written by `python -m predx.cli build`) into the tables the baselines use.

  corpus  : one row per usable device, in predx order (sorted by date, k), with `text` = input_template(ifu, description)
  queries : dev + test subjects with >= 1 gold predicate in the corpus; `positives` = gold_in_corpus (all in-corpus gold,
            primary + additional), `primary` = the primary predicate if it is in the corpus, `n_candidates` = number of
            corpus rows dated strictly before the subject (the candidate set is corpus[:n_candidates])
  pairs   : (q, p, role) train positives, used by the citation-popularity baseline
The dataset directory is given by --dataset, else the PREDX_DATASET environment variable, else data/dataset.
"""
import gzip, json, os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_dir():
    return os.environ.get("PREDX_DATASET") or os.path.join(ROOT, "data", "dataset")


def read_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def input_text(ifu, desc, tpl):
    parts = []
    if ifu:
        parts.append(tpl["ifu_header"] + ifu)
    if desc:
        parts.append(tpl["desc_header"] + desc)
    return tpl["separator"].join(parts).strip()


def load_dataset(d=None, splits=("dev", "test"), panel=None):
    """panel: restrict corpus and queries to one review advisory committee code (e.g. "CV"); n_candidates and the
    positives are recomputed against the restricted corpus, so a quick run on one panel is self-consistent."""
    from bisect import bisect_left
    d = d or default_dir()
    card = json.load(open(os.path.join(d, "dataset_card.json"), encoding="utf-8"))
    tpl = card["config"].get("input_template") or {"ifu_header": "", "desc_header": "", "separator": " "}
    corpus = pd.DataFrame(list(read_jsonl(os.path.join(d, "corpus.jsonl.gz"))))
    if panel:
        corpus = corpus[corpus.panel == panel].reset_index(drop=True)
    corpus["text"] = [input_text(a, b, tpl) for a, b in zip(corpus.ifu, corpus.description)]
    in_corpus = set(corpus.k)
    dates = corpus.date.tolist()
    rows = []
    for s in splits:
        for q in read_jsonl(os.path.join(d, f"queries_{s}.jsonl.gz")):
            if panel and q["panel"] != panel:
                continue
            positives = [p for p in q["gold_in_corpus"] if p in in_corpus]
            if not positives:
                continue  # not retrievable
            primary = [p for p in q["gold_primary_in_corpus"] if p in in_corpus]
            n_cand = bisect_left(dates, q["date"]) if panel else int(q["n_candidates"])
            same = q.get("n_same_code_candidates", 0)
            if panel:
                same = int(((corpus.product_code == q["product_code"]) & (corpus.date < q["date"])).sum())
            rows.append(dict(
                k=q["k"], split=q["split"], date=q["date"], product_code=q["product_code"], panel=q["panel"],
                query=input_text(q["ifu"], q["description"], tpl), positives=positives, primary=primary[0] if primary else None,
                n_candidates=n_cand, n_same_code_candidates=int(same), n_gold=len(q["gold"]),
            ))
    queries = pd.DataFrame(rows)
    pairs = [p for p in read_jsonl(os.path.join(d, "train_pairs.jsonl.gz")) if p["p"] in in_corpus]
    return corpus, queries, pairs, card
