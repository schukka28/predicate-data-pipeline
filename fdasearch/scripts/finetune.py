"""Fine-tune a sentence-transformers embedder on the predx train pairs (subject -> cited predicate).

  python scripts/finetune.py --dataset E:/predx_data/dataset --model google/embeddinggemma-300m --out models/eg300m-ft

Training rows are (anchor = train subject text, positive = one in-corpus gold predicate, negative_1..n = hard negatives).
Hard negatives come from TF-IDF: the top-ranked earlier devices for the subject that its summary does not cite at all
(gold, reference and out-of-corpus citations are all excluded), skipping the first --neg-skip ranks, where an uncited
device is most likely a valid predicate the applicant simply did not name. Loss: cached multiple-negatives ranking
(in-batch negatives + the hard ones), which keeps a large effective batch on one 24 GB GPU.
Only train-split subjects are used; dev is touched only by the small in-training monitor, test never.
Evaluate a checkpoint with:  python scripts/run_baselines.py --dataset ... --pc --methods emb:models/eg300m-ft/final
"""
import argparse, json, os, random, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predx_dataset import load_dataset, read_jsonl
from run_baselines import PROMPTS, prompts_for

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mine_negatives(corpus, tq, cited, n_keep, skip, cache):
    """tq: train queries DataFrame. Returns {query k: [corpus row indices]} of TF-IDF hard negatives."""
    if os.path.exists(cache):
        return json.load(open(cache))
    from sklearn.feature_extraction.text import TfidfVectorizer
    t0 = time.time()
    tv = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.5, stop_words="english", ngram_range=(1, 2))
    D = tv.fit_transform(corpus.text.tolist()); Q = tv.transform(tq["query"].tolist()); DT = D.T.tocsr()
    kidx = {k: i for i, k in enumerate(corpus.k)}
    out = {}
    for qi, q in enumerate(tq.itertuples()):
        s = (Q[qi] @ DT).toarray().ravel()
        s[q.n_candidates:] = -np.inf          # corpus is date-sorted: only devices cleared before the subject
        for c in cited[q.k]:
            if c in kidx:
                s[kidx[c]] = -np.inf
        if q.k in kidx:
            s[kidx[q.k]] = -np.inf
        m = min(skip + n_keep, int(np.isfinite(s).sum()))
        top = np.argpartition(-s, m - 1)[:m] if m > 0 else np.array([], dtype=int)
        top = top[np.argsort(-s[top], kind="stable")][skip:]
        out[q.k] = [int(i) for i in top]
        if qi % 2000 == 0:
            print(f"  mining {qi}/{len(tq)}  ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(out, open(cache, "w"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--model", default="google/embeddinggemma-300m")
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "eg300m-ft"))
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--mini-batch", type=int, default=16, help="chunk size inside the cached loss (memory, not quality)")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--hard-negs", type=int, default=2, help="hard negatives per training row")
    ap.add_argument("--neg-pool", type=int, default=30, help="TF-IDF ranks kept per subject after --neg-skip")
    ap.add_argument("--neg-skip", type=int, default=5)
    ap.add_argument("--monitor-queries", type=int, default=300)
    ap.add_argument("--monitor-docs", type=int, default=8000)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--patience", type=int, default=0, help="stop after this many monitor evals without a new best; "
                    "checkpoints are then saved every --eval-steps and the best one is reloaded at the end")
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0, help="use only this many training rows (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed)

    corpus, q_all, pairs, card = load_dataset(a.dataset, splits=("train", "dev"))
    tq = q_all[q_all.split == "train"].reset_index(drop=True)
    dq = q_all[q_all.split == "dev"].reset_index(drop=True)
    d = a.dataset or os.environ.get("PREDX_DATASET") or os.path.join(ROOT, "data", "dataset")
    cited = {}
    for q in read_jsonl(os.path.join(d, "queries_train.jsonl.gz")):
        cited[q["k"]] = {g["k"] for g in q["gold"]} | set(q.get("reference") or [])
    print(f"corpus {len(corpus)}, train subjects {len(tq)}, dev subjects {len(dq)}", flush=True)

    negs = mine_negatives(corpus, tq, cited, a.neg_pool, a.neg_skip,
                          os.path.join(ROOT, "results", "ft", f"tfidf_negs_{card['built_at'][:10]}_{len(corpus)}_s{a.neg_skip}_n{a.neg_pool}.json"))
    qp, dp = prompts_for(a.model)
    text = corpus.text.tolist(); kidx = {k: i for i, k in enumerate(corpus.k)}
    rows = []
    for q in tq.itertuples():
        pool = negs.get(q.k, [])
        if len(pool) < a.hard_negs:
            continue
        for p in q.positives:
            r = {"anchor": qp + q.query, "positive": dp + text[kidx[p]]}
            for j, ni in enumerate(random.sample(pool, a.hard_negs)):
                r[f"negative_{j + 1}"] = dp + text[ni]
            rows.append(r)
    random.shuffle(rows)
    if a.limit:
        rows = rows[:a.limit]
    print(f"{len(rows)} training rows, {a.hard_negs} hard negatives each; prompts {qp!r} / {dp!r}", flush=True)

    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    model = SentenceTransformer(a.model, device="cuda", trust_remote_code=True)
    model.max_seq_length = a.max_len
    train = Dataset.from_list(rows)
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=a.mini_batch)

    # in-training monitor on dev: a sample of dev subjects against their gold + random earlier devices (a trend, not the benchmark)
    ms = dq.sample(min(a.monitor_queries, len(dq)), random_state=a.seed)
    doc_ids = {kidx[p] for ps in ms.positives for p in ps}
    doc_ids |= set(np.random.default_rng(a.seed).choice(int(ms.n_candidates.min()), a.monitor_docs, replace=False).tolist())
    ev = InformationRetrievalEvaluator(
        queries={q.k: qp + q.query for q in ms.itertuples()},
        corpus={corpus.k[i]: dp + text[i] for i in sorted(doc_ids)},
        relevant_docs={q.k: set(q.positives) for q in ms.itertuples()},
        name="dev-monitor", batch_size=64, mrr_at_k=[10], ndcg_at_k=[10], accuracy_at_k=[1, 10], precision_recall_at_k=[10], map_at_k=[100],
        show_progress_bar=False)

    args = SentenceTransformerTrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs, per_device_train_batch_size=a.batch, learning_rate=a.lr,
        warmup_steps=a.warmup, bf16=torch.cuda.is_bf16_supported(), batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps", eval_steps=a.eval_steps, save_total_limit=12,
        **(dict(save_strategy="steps", save_steps=a.eval_steps, load_best_model_at_end=True, greater_is_better=True,
                metric_for_best_model="eval_dev-monitor_cosine_recall@10") if a.patience else dict(save_strategy="epoch")),
        logging_steps=20, report_to="none", seed=a.seed, dataloader_num_workers=0)
    from transformers import EarlyStoppingCallback
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train, loss=loss, evaluator=ev,
                                         callbacks=[EarlyStoppingCallback(a.patience)] if a.patience else None)
    os.makedirs(a.out, exist_ok=True)
    cfg = {**vars(a), "rows": len(rows), "dataset_built_at": card["built_at"], "corpus": len(corpus)}
    json.dump(cfg, open(os.path.join(a.out, "train_config.json"), "w"), indent=1)
    print("before training:", {k.split("_")[-1]: round(v, 4) for k, v in ev(model).items() if "recall@10" in k or "accuracy@1" in k}, flush=True)
    trainer.train()
    final = os.path.join(a.out, "final")
    model.save_pretrained(final)
    json.dump({**cfg, "best_checkpoint": trainer.state.best_model_checkpoint, "steps": trainer.state.global_step},
              open(os.path.join(a.out, "train_config.json"), "w"), indent=1)
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
