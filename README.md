# predx — 510(k) predicate-retrieval data extraction, audit and conceptual tests

Data pipeline for **"Retrieving 510(k) Predicate Devices with Fine-Tuned Text Embeddings"** (COMS 4705).
It turns FDA 510(k) summary PDFs into a retrieval dataset (query = *Indications for Use* + *Device Description* of a
subject device, gold = the predicates it cites) and then **audits the dataset against the reviewer's questions**
before any model is trained. It is built to run on a Google Cloud (Compute Engine) VM and is resumable after a Spot
preemption.

```
openFDA metadata ─▶ plan ─▶ fetch PDFs ─▶ text (pdftotext, OCR fallback) ─▶ parse sections + predicates
      ─▶ build (corpus / queries / splits / train pairs / leakage masking) ─▶ validate ─▶ audit (PASS/WARN/FAIL report)
```

## 0. Ten-minute start (no network needed)

```bash
pip install -r requirements.txt            # plus: apt install poppler-utils tesseract-ocr   (only needed for real PDFs)
python -m pytest tests -q                  # 72 tests (+1 that runs on real PDFs you drop in tests/real_pdfs/), ~1.5 min
python -m predx.cli --config config/demo.yaml demo           # synthetic world, ~5 s -> data_demo/reports/audit.md
python -m predx.cli --config config/demo.yaml demo --tiny    # 60 devices: SHOULD fail the size checks
python -m predx.cli --config config/demo.yaml demo --with-pdfs 12   # 12 docs rendered as real PDFs (1/3 scanned) through pdftotext/OCR
```

The **demo is the "small conceptual test"**: `predx/synth.py` builds a fake 510(k) world with a *planted* citation graph
(roles, three predicate-list styles, unknown/PMA/DEN/future citations, leakage, missing PDFs, scanned pages) and records the
truth in `truth.json`, so the tests can check that extraction recovers exactly what was planted. Its audit is *meant* to
show WARN/FAIL lines (synthetic text is long, `--tiny` is too small); the point is to see what each failure looks like
before you meet it on real data. **Synthetic numbers are not results.**

## 0b. Testing on real PDFs (the biggest risk) — do this first on the VM

The sandbox this was built in cannot reach `accessdata.fda.gov`, so the real-PDF path was tested on **synthetic PDFs** (digital and
scanned, with planted answers), not FDA files. Three tools close that gap quickly:

```bash
# 1. Download a few real PDFs and see exactly what the parser makes of them (needs network: run on the VM)
python -m predx.cli probe --ks-file config/probe_ks.txt          # edit the list: ~15 K-numbers, deliberately varied
python -m predx.cli probe K243175 K191717                        # or name them directly
python -m predx.cli probe --pdf-dir ~/my_pdfs                    # or PDFs you already have (file name must contain the K-number)
#    -> data_probe/probe_report.md : per page type/OCR/first line, heading candidates, section START *and END*, predicates with
#       roles + snippets, every line that mentions predicates, and warnings (NO_IFU, DESC_CAPPED, MOSTLY_OCR, ...)
#       data_probe/text/<K>.txt    : the full extracted text, page by page

# 2. Turn hand-verified documents into regression tests
cp *.pdf tests/real_pdfs/ ; python -m pytest tests/test_real_pdfs.py -s        # see tests/real_pdfs/README.md (expected.json)

# 3. How badly do scans hurt? (synthetic scans with planted truth; ~3 min on 2 vCPUs)
python -m predx.cli ocr-stress --n 8 --levels 0,1,2,3
```

**What the OCR stress test found** (16 synthetic documents per level, planted predicates; level 0 = clean render, 1 = ordinary
bad copy, 2 = poor, 3 = unrealistically bad, about 84 dpi effective). "before" = tesseract defaults; "after" = the current defaults
(median-filter pre-processing + K-number digit re-read):

| level | indications recovered | description recovered | all K-numbers exact | primary correct | gold set exact | gold recall / precision | s per doc |
|---|---|---|---|---|---|---|---|
| 0 clean | 100% | 100% | 100% | 100% | 100% | 100% / 100% | 5.6 |
| 1 before | 56% | 38% | 100% | 94% | 81% | 94% / 97% | 7.3 |
| 1 after | **100%** | **100%** | 100% | 100% | **100%** | 100% / 100% | 10.4 |
| 2 before | 0% | 0% | 50% | 75% | 44% | 76% / 93% | 7.4 |
| 2 after | **75%** | **88%** | **88%** | **88%** | **88%** | **94%** / 94% | 10.9 |
| 3 before | 0% | 0% | 0% | 25% | 0% | 15% / 83% | 31.0 |
| 3 after | 0% | 0% | 25% | 44% | 25% | 52% / 71% | 19.8 |

With only 16 documents per cell a single document is 6 points: treat differences under ~12 points as noise. The pattern is consistent across every metric, though.

* The fix that mattered: a **3x3 median filter + autocontrast before tesseract** (`text.ocr.preprocess: median_gray`, default). I also tried
  explicit binarisation, adaptive thresholds, other page-segmentation modes and higher DPI. Binarisation cleans prose but turns `0`/`9` into `6`
  (`K250001` becomes `K250061`) and **tesseract's own confidence rises while it does so**, so confidence cannot be used to detect it. Pre-processing options are `none`, `median_gray` (default) and `median_otsu`.
  psm 6, 400 dpi and adaptive thresholding were all worse. Deskewing added almost nothing.
* Second fix: **`text.ocr.k_reread`** re-reads every K-number-looking token from an upscaled crop (2x/3x/4x) with a digits-only whitelist and takes a majority vote with the page reading. On poor scans (level 2, same documents, pre-processing on in both): all K-numbers exact 69% -> 88%, gold set exact 69% -> 88%, primary 75% -> 88%; no change on scans that were already fine; about +20% OCR time.
* On poor scans the remaining failure is **digit errors inside K-numbers**, i.e. wrong labels, not missing text. A misread K-number is usually an unknown number
  (it ends up as `not_in_openfda`, lowering the recall ceiling), but about a third of the 6-digit K-number space is occupied, so a misread can also land on a
  *different real device*. We do not "repair" digit-for-digit confusions for that reason (only letter-for-digit ones like `O`/`l`/`S`, and only when the repaired number is a real device).
* **What to do with this on real data:** run the probe on 3–5 scanned (pre-2012) PDFs and compare the K-numbers by eye with the PDF; check the audit's OCR share and
  OCR-confidence line (§2); make sure the 100-document hand-labelled sample contains OCR'd documents (it flags them) and look at precision on those separately.
  If OCR'd documents turn out to be a large share of the corpus or their labels poor, compare `--set text.ocr.preprocess=none` and consider dropping their predicates.
* These curves are for synthetic scans of a monospace font. Real FDA scans have other artefacts (stamps, handwriting, tables, fax noise), so treat the numbers as a regression
  guard and a ranking of OCR options, **not** as expected accuracy.

## 0c. How much gold data will there be? (`predx sizing`, `docs/DATA_SIZING.md`)

```bash
python -m predx.cli sizing        # planning model; after `meta` it recomputes the counts from your own openFDA download
python -m predx.cli sizing --set sizing.scenarios.mid.pred_rate=0.85 --set sizing.scenarios.mid.usable_subject=0.9   # plug in pilot rates
```

Real openFDA counts (retrieved 2026-11-19) give **34,782 summaries for 2015-2026: 25,987 train subjects (2015-2023), 2,935 dev (2024), 5,860 test (2025-2026)**.
With assumed extraction yields the mid case is about **20k train queries / 31k train pairs, 2.3k dev queries, 4.5k test queries** (low-high: 15.5k-23.4k train queries),
a candidate pool of 51-59k devices in 2015 growing to 79-91k, and a recall ceiling of about 80-85% of cited predicates. The yields are assumptions until the pilot audit
replaces them. `docs/DATA_SIZING.md` has the full tables, the data structure, the published numbers it uses (with links) and the feasibility verdict.

## 1. Reviewer feedback → what implements it

| Reviewer comment | Where it is handled | Where to read the answer |
|---|---|---|
| Prioritize the extraction pilot early (PDF/OCR surprises) | `run_pilot.sh` / `predx pilot`: one specialty (default panel RA), ~400 subjects **plus the predicates they cite**; per-page OCR trigger, page counts, OCR share, fetch-folder patterns, seconds/document are logged | audit §2 *Extraction health*, §9 *Projection* |
| Token lengths of the two input sections; truncation / chunking across models | `tokens.py` loads each model's own tokenizer; grid of max_len (128–1024); `head` vs `budget` policy (reserves tokens for the indications); the prompts add overhead; chunk-and-max-pool fallback sized from the same data | audit §3 (per model: percentiles, % truncated, decision rule) |
| How many cited predicates have usable records (recall ceiling) | every gold predicate gets `in_corpus` + a **reason** when missing (not_in_openfda, no_pdf_404, statement_only, temporal_violation, non_510k_identifier, …) | audit §4 (by reason, split, predicate year, "ceiling if the corpus started in year Y") |
| Specify the similarity function; keep it consistent; allow dot-product | `eval_spec` in config → copied to `dataset_card.json`; `metrics.score_matrix` does cosine on L2-normalised vectors (primary) or dot (ablation) | `config/default.yaml`, `predx/metrics.py` |
| Hybrid BM25 + embedding later | sparse BM25 baseline over the same candidate sets; same ranking interface so a fusion step can slot in | audit §8, `predx/baselines.py` |
| Un-fine-tuned baseline for each model; architecture/size confound | `zero_shot_baseline: true` per model; commented matched-architecture control (`gte-modernbert-base`) in `models:` | `config/default.yaml` |
| Recall@1, @5, @10 | `eval_spec.recall_k`, `hit_k`, MRR, nDCG@10, bootstrap 95% CI (all implemented in `metrics.py`, used by the baselines) | audit §8 |
| Candidates and cited predicates per query; why k = 10 | distributions of candidates, gold and gold-in-corpus per query; same-product-code candidates; for each k, the % of queries with more than k gold predicates and the best Recall@k a perfect ranker could reach | audit §5 |
| Frame results as recovery of historical citations | stated in the report header, `eval_spec.report_as` and the dataset card; `retrievable` flag marks queries with ≥1 gold in the corpus | audit header, §4–5 |
| "Enough examples?" | split sizes, 95% CI half-width at the assumed Recall@10, slices (prior same-code clearances, seen/unseen predicates), training-popularity skew | audit §6 |
| (not asked, but fatal if wrong) label leakage | K-numbers and predicate names masked in query text; residual leakage measured | audit §7 |

## 2. Run order (milestone is Mon Nov 23)

1. **Tonight (laptop or VM):** `pytest` + the demo, read `data_demo/reports/audit.md` once so the report is familiar.
2. **Fri — real PDFs on a GCE VM** (§3): `./scripts/run_probe.sh` (5 min; §0b), fix what it shows, then `./scripts/run_pilot.sh` and read `data/reports_pilot/audit.md`.
   Whatever breaks here (URL patterns, scanned PDFs, headings the parser misses) is the whole point of the pilot.
3. **Fri/Sat — hand-check 100 summaries:** `python -m predx.cli --config config/pilot.yaml label-sample`, open
   `data/labeling/label_sample.csv`, fill `hand_primary / hand_additional / hand_reference` (K-numbers separated by `;`)
   and `done=y`, then `label-eval`. It reports gold-set precision/recall/F1, primary-predicate accuracy (vs the naive first-other-number rule) and
   exact-set match per subject, with Wilson intervals, plus reference-device precision/recall and the first misses. Fix rules in `sections.py` / `predicates.py`, run `python -m predx.cli parse`
   (cheap, no OCR re-run) and `label-eval` again. This is the "100 hand-labeled summaries" check from the proposal.
4. **Sat/Sun — full run:** `AUTORESUME=1 SHUTDOWN_WHEN_DONE=1 ./scripts/run_full.sh` (inside tmux). Audit §9 gives the projected time/disk from
   the pilot; the download alone is bounded by `fetch.requests_per_second` (4/s ⇒ 60k PDFs ≈ 4.2 h).
5. **Mon:** the audit tables (§1 funnel, §3 token lengths, §4 recall ceiling, §5 candidates/gold per query, §6 sufficiency)
   are the data-section content for the milestone.

## 3. Google Compute Engine

**No cloud credits yet?** `notebooks/predx_colab_test.ipynb` (open it in Colab via File → Open notebook → GitHub, then set `REPO_GIT_URL` in the first cell) runs the whole test ladder (unit tests, synthetic ETL, 10 real PDFs, OCR stress, live FDA probe, mini pilot, hand check) on free Google Colab.

**Shared Google Cloud project, authorising teammates, results in a bucket, reruns against the same data: [`docs/GCP_SHARED_RUNBOOK.md`](docs/GCP_SHARED_RUNBOOK.md).**

**Full run, step by step (settings, VM, monitoring, judging, fixing): [`docs/RUNBOOK_FULL.md`](docs/RUNBOOK_FULL.md).**

**Shared team project, probe/pilot step by step, and how to judge them: [`docs/GCP_TEAM.md`](docs/GCP_TEAM.md)** (`scripts/gcp_team_setup.sh`, `scripts/check_run.sh`).

From your laptop / Cloud Shell (needs `gcloud auth login`):

```bash
PROJECT=my-project BUCKET=my-unique-bucket ./scripts/gcp_create_vm.sh      # Spot e2-standard-8, 300 GB pd-balanced, + results bucket
gcloud compute scp --recurse --zone=us-central1-a ./predicate-data predx-extract:~/
gcloud compute ssh predx-extract --zone=us-central1-a
```

On the VM:

```bash
cd ~/predicate-data
CONTACT_EMAIL=you@example.com GCS_BUCKET=my-unique-bucket HF_TOKEN=hf_xxx ./scripts/vm_setup.sh   # apt + venv + tests + demo
tmux new -s predx
./scripts/run_pilot.sh            # PANEL=RA N=400 by default
./scripts/run_full.sh             # later; resumable
./scripts/sync_gcs.sh up          # copy results to the bucket (run_full.sh also does this every 10 min)
```

* **Spot preemption:** the pipeline writes atomically and logs append-only, so re-running skips finished work. A stopped Spot VM does *not*
  restart itself: `gcloud compute instances start predx-extract --zone=…`. If you launched with `AUTORESUME=1`, `vm_startup.sh` restarts the job in tmux on boot.
* **Cost:** compute is billed only while the VM runs; `SHUTDOWN_WHEN_DONE=1` powers it off after the audit (disk keeps billing a few $/month).
  Your shared budget is ~$100 including GPU fine-tuning, so stop or delete the VM when extraction is finished.
* **Disk:** PDFs dominate. Set `--set fetch.keep_pdfs=false` to delete each PDF after its text is cached (the text cache is what later stages use).
* **CPU:** OCR is CPU-bound and runs in a process pool (`text.workers`, 0 = all cores). More vCPUs ≈ proportionally less OCR wall time.
* **Hugging Face:** EmbeddingGemma is gated. Accept its licence on the model page and set `HF_TOKEN` or the audit falls back to an *approximate* tokenizer and says so.

## 4. Reading the audit (`data/reports/audit.md`, plus `audit.json` and `tables/*.csv`)

Section 0 lists every check as PASS / WARN / FAIL with value and threshold (thresholds live in `config/default.yaml → checks`; a
metric below `fail_fraction` of its threshold is FAIL, otherwise WARN). Things to look for first:

* **§2:** fetch success by year and folder pattern; share of documents needing OCR; pages per document; extraction seconds/document.
* **§3:** per model, % of descriptions truncated at each max_len. If the 256-token budget truncates >5% of descriptions the report recommends a larger max_len or chunking.
* **§4:** the *recall ceiling* = share of (subject, predicate) pairs whose predicate is in the searchable corpus, and why the rest are missing. If a large share is "predicate older than corpus start / no PDF", decide whether to extend the corpus or just report the ceiling.
* **§5:** candidates per query grow with date (the candidate rule is *decision date strictly before the subject's*). The k-analysis shows, per k, how many queries have more gold predicates than k and the maximum Recall@k a perfect ranker could reach.
* **§6:** whether the test split is big enough for the CI you want, and whether each slice has ≥ `min_slice_n` queries.

### Dataset files (`data/dataset/`)

| file | content |
|---|---|
| `corpus.jsonl.gz` | one row per usable searchable device (`k, date, product_code, panel, device_name, applicant, ifu, description, n_pages`), sorted by (date, k). Candidates for a query = corpus rows `[0, n_candidates)`. |
| `queries_{train,dev,test}.jsonl.gz` | `k, date, ifu, description, gold=[{k, role, in_corpus, reason, year, product_code}], gold_in_corpus, gold_primary_in_corpus, reference, n_candidates, n_same_code_candidates, retrievable, leak` |
| `train_pairs.jsonl.gz` | `{"q": subject, "p": predicate, "role": …}` — only pairs whose predicate is in the corpus |
| `device_status.jsonl.gz` | one row per device with the funnel status/reason (why each device was or was not used) |
| `dataset_card.json` | counts, mask level, and the full config (similarity, k values, prompts, max lengths) so modelling code stays consistent |

## 5. Design decisions worth knowing

* **Splits** are by the *subject's* decision date: train 2015–2023, dev 2024, test 2025–2026. Predicates may be older than the split (that is the task), but never newer than the subject.
* **Predicate extraction** uses a K/BK/DEN/P number finder (whitespace-tolerant, OCR-repair only if the repaired number is a known device), role hints from the nearest label ("Predicate Device", "Reference Device", "Additional predicate"), and the first-other-number rule of Wu et al. as a fallback. Reference devices are stored separately and are **not** treated as gold predicates by default.
* **Input text never contains the answer key.** Sections end at the next heading, so predicate/comparison sections are excluded; standard masking replaces submission numbers with `[K-NUMBER]` and gold-predicate names with `[DEVICE]`; `strict` also masks distinctive name tokens and drops predicate/SE sentences. Switch with `--mask-level`; leakage is measured on the raw text and on the masked text.
* **PDF text:** `pdftotext -layout` per page, per-page OCR (`pdftoppm` + `tesseract`) when a page has fewer than 60 alphanumerics or looks garbled (`(cid:`). Text is cached per page, so parsing rules can be iterated without repeating OCR. `--config config/pymupdf.yaml` switches to the GPU backend (`text.backend: pymupdf`: PyMuPDF text layer + RapidOCR on ONNX Runtime CUDA, no poppler/tesseract; 0.65 s per OCR'd page on an RTX 4090, same cache format).
* **Everything is configurable** in `config/default.yaml`; `--set a.b=c` overrides one value; `--config` layers files.

## 6. Known limits — read before trusting a number

* **No live FDA access was available while building this** (the sandbox's network policy blocks `accessdata.fda.gov`). Fetching, retries and `%PDF` checks were tested against local fake HTTP servers; PDF extraction against synthetic PDFs (digital, and scanned at four degradation levels). Section-heading patterns were checked against the headings of two real 510(k) summaries I could read online, not against a corpus of real PDFs. **Run `probe` first, then the pilot and the 100-document hand check: they are the real test.** Running the real PDF path on synthetic PDFs already exposed one bug (a heading regex with a broken character class that silently disabled "shelf-life" headings) and the OCR weakness above, so expect real PDFs to expose more.
* **FDA may refuse or throttle the downloader** (the User-Agent is a polite academic one with your `CONTACT_EMAIL`). `probe` and the fetch log show HTTP 403/429; if that happens, lower `fetch.requests_per_second` rather than disguising the client.
* **Older PDF URLs are unverified.** Modern summaries live at `…/cdrh_docs/pdf{YY}/{K}.pdf`; for pre-2010 numbers the fetcher tries `pdf{Y}`, `pdf{YY}` and `pdf` in turn and logs which worked (audit §2 shows the pattern counts). Check that table after the pilot if you extend `min_decision_date` backwards.
* **A closure-only corpus leaks labels.** If you only fetch subjects and the predicates they cite, "is it in the corpus?" is itself a signal. The `closure` stage exists to make the pilot self-contained; the **final corpus needs the `full` stage** (every summary-bearing SE device). Do not report retrieval numbers from a pilot.
* **Token counts are approximate** until the VM has `transformers` and (for EmbeddingGemma) an `HF_TOKEN`; the report labels APPROX loudly and WARNs.
* **BM25 / popularity baselines are wiring checks,** not paper results. They exist so the audit can tell you "this data is retrievable at all" and catch date/label bugs.
* The Li et al. predicate-extraction code could not be located, so the extraction rules are reimplemented from the paper descriptions; compare to it only qualitatively.
* openFDA is refreshed regularly; the audit WARNs when its counts drift more than 5% from the proposal's (176,529 decisions / 94,143 with summaries / ~34.6k subjects 2015–2026).

## 7. Layout

```
config/    default.yaml  pilot.yaml  demo.yaml  probe_ks.txt
docs/      DATA_SIZING.md
predx/     cli.py  probe.py  stress.py  sizing.py  metadata.py  plan.py  fetch.py  pdftext.py  text_stage.py  sections.py  knumbers.py  predicates.py
           parse_stage.py  leakage.py  dataset.py  tokens.py  metrics.py  baselines.py  audit.py  labeling.py  synth.py  synth_pdf.py
scripts/   gcp_create_vm.sh  vm_setup.sh  vm_startup.sh  run_probe.sh  run_pilot.sh  run_full.sh  sync_gcs.sh
tests/     knumbers, sections, predicates, leakage, tokens/metrics, PDF round-trip, probe + OCR stress, fetch, synthetic pipeline, labeling, pilot end-to-end
           real_pdfs/ (drop real PDFs here; see its README)
```

## fdasearch/ — zero-shot retrieval baselines and a Drive-backed Colab

`fdasearch/` runs on top of a `predx` build: 16 untuned retrieval baselines (TF-IDF, BM25, LSA, nine dense
embedders, Gemini, RRF hybrids) scored with bootstrap CIs on every dev and test query, a report by review panel and
candidate-pool size, and a Colab that reruns them from a data bundle on Google Drive. It also contributes the
`pymupdf` text backend (`predx/pdftext_pymupdf.py`, `config/pymupdf.yaml`) and `fdasearch/scripts/predx_bridge.py`,
which drops already-downloaded PDFs into a predx data root. See `fdasearch/README.md` and `fdasearch/results/REPORT.md`.
