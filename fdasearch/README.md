# fdasearch: zero-shot baselines for 510(k) predicate retrieval

Task: given a new (subject) device's Indications for Use + Device Description, rank earlier 510(k)
clearances so that the predicates its summary cites come first. The dataset is built by the team's `predx`
pipeline (this repository); `fdasearch/` adds a GPU text-extraction backend for predx, the zero-shot retrieval
baselines with confidence intervals, and a Colab notebook that reruns them from a data bundle on Google Drive.
Numbers are in `results/REPORT.md`.

## How the pieces fit

```
predx (repo root)                       fdasearch/ (this folder)
  meta  -> data/meta/510k.jsonl.gz
  plan, fetch -> data/pdf/<yy>/K.pdf      scripts/predx_bridge.py   put already-downloaded PDFs into the predx root
  text  -> data/text/<yy>/K.txt.gz        predx/pdftext_pymupdf.py  backend "pymupdf": PyMuPDF + RapidOCR on GPU (config/pymupdf.yaml)
  parse -> data/parsed.jsonl.gz
  build -> data/dataset/*.jsonl.gz        scripts/predx_dataset.py  reads corpus / queries_* / train_pairs / dataset_card
  validate, audit -> data/reports/        scripts/run_baselines.py  16 methods, --pc variant, per-query ranks, top-100 lists
                                          scripts/summarize_results.py  REPORT.md with bootstrap CIs, by panel, by pool size
                                          scripts/make_colab.py     notebooks/fdasearch_510k_baselines.ipynb
                                          bundle_build.sh           Drive bundle fdasearch_data_v3 (a predx build + manifest)
```

Definitions follow `config/default.yaml`: train 2015-2023 / dev 2024 / test 2025+, query text = Indications for
Use + Device Description (no device name), K-numbers and cited predicate names masked (`mask_level: standard`),
gold = primary + additional predicates (predx's first-other fallback included), candidates = corpus devices dated
strictly before the subject (`n_candidates`), metrics over all in-corpus gold with primary-only columns as a
secondary view.

## Running it

From the repository root, with a Python that has the predx requirements plus `pymupdf rapidocr-onnxruntime
onnxruntime-gpu bm25s PyStemmer scikit-learn sentence-transformers google-genai`:

```
# 1. data: either predx's own fetch (python -m predx.cli run-all) or PDFs you already have
python fdasearch/scripts/predx_bridge.py --predx-root E:/predx_data --pdf-dir /path/to/pdfs
python -m predx.cli --config config/pymupdf.yaml --set paths.data_dir=E:/predx_data text --queue all
python -m predx.cli --set paths.data_dir=E:/predx_data parse
python -m predx.cli --set paths.data_dir=E:/predx_data build
python -m predx.cli --set paths.data_dir=E:/predx_data validate
python -m predx.cli --set paths.data_dir=E:/predx_data audit

# 2. baselines (results land in fdasearch/results/)
python fdasearch/scripts/run_baselines.py --dataset E:/predx_data/dataset --pc --methods random,pop,bm25,tfidf,lsa
python fdasearch/scripts/run_baselines.py --dataset E:/predx_data/dataset --pc --methods emb:BAAI/bge-m3,ollama:embeddinggemma,gemini:gemini-embedding-001
python fdasearch/scripts/run_baselines.py --dataset E:/predx_data/dataset --methods "rrf:tfidf|gemini_gemini-embedding-001"
python fdasearch/scripts/summarize_results.py --dataset E:/predx_data/dataset

# 3. notebook + bundle
python fdasearch/scripts/make_colab.py
bash fdasearch/bundle_build.sh E:/predx_data
```

`--panel CV` on `run_baselines.py` and `summarize_results.py` restricts corpus and queries to one review panel
(candidate counts are recomputed), which is what the notebook's quick mode uses. Method specs:
`bm25`, `tfidf`, `lsa`, `pop`, `random`, `emb:<hf model>`, `ollama:<model>`, `gemini:<model>`, `rrf:a|b` (names as
saved under `results/top/`). Embeddings are cached per K-number in `results/emb/`.

### The pymupdf text backend

`text.backend: pymupdf` (set by `config/pymupdf.yaml`) replaces poppler + tesseract with PyMuPDF for the text layer
and RapidOCR (ONNX Runtime, CUDA when available) for image-only or garbled pages, same on-disk format and
`text_stats.jsonl` fields. Measured on an RTX 4090 at 150 dpi: 0.65 s per OCR'd page, the whole 49.5k-PDF corpus in
about 50 minutes with 8 workers. The recognizer is the English PP-OCRv3 model from the Hugging Face hub
(`ocr.rec_model`), because RapidOCR's bundled Chinese+English model drops the spaces between English words and
the heading rules need them. It has no K-number digit re-read vote (`k_reread_changed` is always 0).

## Results

See `results/REPORT.md` (test split, all in-corpus gold, bootstrap 95% CIs, breakdowns by panel and by candidate
pool size) and `results/metrics.csv`. Earlier runs on fdasearch's own pipeline are kept for comparison in
`results_v1/` (2010+ corpus, no OCR) and `results_v2/` (closure fetch + OCR + SE-sentence stripping); the ranking of
methods was the same in both: TF-IDF first, reciprocal-rank fusion with Gemini or EmbeddingGemma on par, BM25 and
the untuned embedders 4-7 points behind on Recall@10.

## Data bundle for the Colab (hermetic w.r.t. data)

`bundle_build.sh` writes `bundle_v3/` (mirrored to the Drive folder `fdasearch_data_v3`): the predx build
(`dataset/`), `parsed.jsonl.gz`, `text_stats.jsonl`, `fetch_status.jsonl` and `meta/510k.jsonl.gz` so `predx build`
can be re-run with another mask level, `reports/audit.md`, the reference `metrics.csv` / `REPORT.md`, and
`manifest.json` with a sha256 per file that the notebook verifies. Models are still pulled from Hugging Face / the
Gemini API and embedded in Colab.

## Other scripts

- `scripts/parse_agreement.py` + `results/parse_agreement.md`: the check that led to adopting the predx parser
  (same primary predicate on 95.9% of 3,089 sampled documents; closer to the LLM labels). It reads fdasearch's
  retired v2 extraction, so it is kept as a record rather than as a tool.
- `scripts/llm_label_check.py` (run on a Spark with Ollama) asks gemma4:31b for predicate/reference K-numbers on
  100 random 2024-2026 summaries; output in `results/label_check_llm.jsonl`.

## Metrics

Recall@10 (fraction of a subject's in-corpus gold predicates in the top 10), Hit@1/Hit@10/Hit@100 (any gold in
the top k), MRR, and the same for the primary predicate only. Gold excluded by a filter (`+pc`) counts as not
retrieved. CIs are bootstrap over queries (1000 resamples).

## Known label noise

On the 100-summary LLM check, the predx gold set has precision 0.71 / recall 0.95 against gemma4:31b's reading
(over-inclusion is mostly reference devices and the subject's own prior clearances). All scores therefore
underestimate usefulness; primary-only metrics are the cleanest.

## Environment

Local: RTX 4090, `.venv` with CUDA torch (2.11+cu128), sentence-transformers, bm25s, rapidocr-onnxruntime with
onnxruntime-gpu 1.22 (import torch before RapidOCR so the CUDA DLLs load; the backend does this). Gemini embeddings
need `GEMINI_API_KEY` in `fdasearch/.env`. EmbeddingGemma is served from Ollama (`embeddinggemma`) because the
Hugging Face weights are gated; nomic-embed-text also via Ollama. Sparks (`ssh spark`, `ssh spark2`) have Ollama
but no torch.
