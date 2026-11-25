"""Assemble notebooks/fdasearch_510k_baselines.ipynb: a Colab that clones predicate-data-pipeline (the team's predx
pipeline plus fdasearch/), gets the data either from the Drive bundle (a predx build) or by running the predx stages
against FDA, and runs the zero-shot retrieval baselines. Re-run after changing the scripts or when new results land
in results/metrics.csv (the intro table is read from there)."""
import os, json, textwrap
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebooks", "fdasearch_510k_baselines.ipynb")


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(src).strip("\n")}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": textwrap.dedent(src).strip("\n")}


def results_table():
    """Top methods on the test split from the local full run, for the intro cell."""
    f = os.path.join(ROOT, "results", "metrics.csv")
    if not os.path.exists(f):
        return "(results/metrics.csv not found when the notebook was generated)"
    m = pd.read_csv(f).drop_duplicates(["method", "split"], keep="last")
    m = m[(m.split == "test") & ~m.method.str.endswith("+pc")].sort_values("recall10", ascending=False)
    show = ["tfidf", "rrf:tfidf|gemini_gemini-embedding-001", "rrf:tfidf|ollama_embeddinggemma", "bm25",
            "gemini:gemini-embedding-001", "ollama:embeddinggemma", "emb:BAAI/bge-m3", "emb:intfloat/e5-base-v2",
            "emb:Qwen/Qwen3-Embedding-4B", "lsa", "pop"]
    nice = {"tfidf": "TF-IDF (1-2 grams, cosine)", "bm25": "BM25", "lsa": "LSA (SVD-300)", "pop": "product-code popularity",
            "ollama:embeddinggemma": "EmbeddingGemma-300M", "gemini:gemini-embedding-001": "Gemini embedding-001",
            "rrf:tfidf|ollama_embeddinggemma": "RRF TF-IDF + EmbeddingGemma", "rrf:tfidf|gemini_gemini-embedding-001": "RRF TF-IDF + Gemini",
            "emb:BAAI/bge-m3": "bge-m3", "emb:intfloat/e5-base-v2": "e5-base-v2", "emb:Qwen/Qwen3-Embedding-4B": "Qwen3-Embedding-4B"}
    rows = ["| method | Recall@10 | Hit@1 | Hit@10 | Hit@100 | MRR |", "|---|---|---|---|---|---|"]
    for meth in show:
        r = m[m.method == meth]
        if len(r):
            r = r.iloc[0]
            rows.append(f"| {nice.get(meth, meth)} | {100*r.recall10:.1f} | {100*r.hit1:.1f} | {100*r.hit10:.1f} | {100*r.hit100:.1f} | {r.mrr:.2f} |")
    n = int(m.n.iloc[0])
    return f"Test split (n={n} subjects cleared 2025-2026), no fine-tuning:\n\n" + "\n".join(rows)


cells = []
cells.append(md(f"""
# Zero-shot predicate retrieval for FDA 510(k) devices

Given a new device's **Indications for Use + Device Description**, rank earlier 510(k) clearances so that the
predicates its summary cites come first. This notebook runs the untuned retrieval baselines (lexical, dense
embeddings, hybrids) on the dataset built by the team's `predx` pipeline. COMS 4705 project: *Retrieving 510(k)
Predicate Devices with Fine-Tuned Text Embeddings* (K. Azami, J. Lamm, S. Chukka).

**Pipeline** (`predx`, github.com/schukka28/predicate-data-pipeline; every stage is resumable)

1. `meta`: openFDA 510(k) metadata (176k records); `plan` + `fetch`: summary PDFs from accessdata.fda.gov
2. `text`: per-page text. Backend `pymupdf` = PyMuPDF text layer + RapidOCR on the GPU for image-only pages
   (`config/pymupdf.yaml`); the default backend is poppler + tesseract
3. `parse`: section headings (Indications for Use, Device Description) and cited K-numbers with predicate /
   reference roles; `build`: K-numbers and predicate names masked, train 2015-2023 / dev 2024 / test 2025+,
   gold = primary + additional predicates present in the corpus; `validate` + `audit`
4. `fdasearch/scripts/run_baselines.py`: baselines + metrics (Recall@10, Hit@k, MRR with bootstrap CIs; primary-only
   columns as a secondary view); `summarize_results.py`: report by review panel and candidate-pool size

**Data source.** `DATA_SOURCE = "drive"` (default) is hermetic with respect to data: it reads the prebuilt bundle
(`fdasearch_data_v3` on Google Drive: the predx build `data/dataset/`, the parsed records so `build` can be re-run
with another mask level, the audit report and the reference metrics; every file sha256-verified against
`manifest.json`). Nothing is fetched from FDA and the dataset is byte-identical to the reference run. Models are still
downloaded from Hugging Face and embedded on the Colab GPU (Gemini via the API). `DATA_SOURCE = "fda"` runs the
predx stages against openFDA and accessdata.fda.gov instead.

**Two modes.** `quick` restricts everything to one advisory committee (default Cardiovascular; a few minutes on a T4
from the bundle; with `fda` it runs `predx pilot`, a stratified sample of `PILOT_N` subjects plus the predicates they
cite, about 20 minutes). `full` uses all panels (~49k documents; lexical methods in minutes, each dense model 20-40
minutes on a T4, hours more when fetching from FDA).

**Reference results from the full run on an RTX 4090** (same code; numbers in a quick run differ because the
candidate pool is one panel):

{results_table()}

Lexical search wins zero-shot (TF-IDF, then BM25); the untuned embedders (Qwen3, bge-m3, EmbeddingGemma) sit 9-10 points
lower on Recall@10; rank fusion with TF-IDF matches TF-IDF on Hit@100 but not on Recall@10. Gemini was not run on this build.
About 9% of the regex "primary predicate" labels disagree with an LLM read of the summary, so all scores are lower bounds.
"""))

cells.append(md("## 0. Settings"))
cells.append(code("""
REPO_URL   = "https://github.com/keyvanazami/predicate-data-pipeline"   # predx + fdasearch/ (fork branch; use schukka28/... main once merged)
REPO_REF   = "fdasearch-predx"                                           # branch, tag or commit to check out

DATA_SOURCE = "drive"       # "drive" = prebuilt predx build from Google Drive (hermetic); "fda" = run the predx stages against FDA
DATA_DIR    = "/content/drive/MyDrive/fdasearch_data_v3"   # bundle location after drive.mount (if it was shared with you: Drive > Organize > Add shortcut to My Drive)
DATA_FOLDER_ID = ""                                         # fallback: fetch the folder by id with gdown (needs "anyone with the link" sharing)
MASK_LEVEL  = None          # None = use the shipped build; "standard" | "strict" | "none" = re-run `predx build` from the parsed records

MODE       = "quick"        # "quick" = one advisory committee; "full" = all panels
PANEL      = "CV"           # quick mode only. Codes: OR Orthopedic, RA Radiology, CV Cardiovascular, SU Surgery, DE Dental, HO Hospital, GU Gastro/Urology, NE Neurology, ...
PILOT_N    = 400            # fda + quick: number of subjects in the pilot sample (predx plan --stage pilot)

# Dense models, downloaded from Hugging Face and run on the Colab GPU. EmbeddingGemma is gated: accept the license at
# https://huggingface.co/google/embeddinggemma-300m and put a token in Colab secrets as HF_TOKEN.
EMBED_MODELS = ["google/embeddinggemma-300m", "BAAI/bge-m3"]      # [] to skip dense models
RUN_GEMINI   = True         # Gemini embedding-001 via the API; needs GEMINI_API_KEY in Colab secrets (skipped if absent)

import os
def secret(name):
    try:
        from google.colab import userdata
        v = userdata.get(name)
        if v: return v
    except Exception:
        pass
    return os.environ.get(name, "")
HF_TOKEN = secret("HF_TOKEN"); GEMINI_API_KEY = secret("GEMINI_API_KEY")
print("HF token:", "yes" if HF_TOKEN else "no (EmbeddingGemma will be skipped)", "| Gemini key:", "yes" if GEMINI_API_KEY else "no (skipped)")
"""))

cells.append(md("## 1. Environment"))
cells.append(code("""
%pip install -q "pymupdf>=1.24" "pandas>=2.2" "bm25s>=0.2" PyStemmer "scikit-learn>=1.4" "sentence-transformers>=5.0" "transformers>=4.56" google-genai requests pyyaml pypdf
if DATA_SOURCE == "fda":
    %pip install -q rapidocr-onnxruntime onnxruntime-gpu
import os, sys, json, glob, subprocess, pathlib
def sh(cmd):
    # run a shell command and stream its output into the cell (Colab does not reliably show inherited stdio)
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         env={**os.environ, "PYTHONUNBUFFERED": "1"})
    tail = []
    for line in p.stdout:
        print(line, end="", flush=True); tail.append(line); tail = tail[-40:]
    if p.wait():
        raise RuntimeError(f"command failed ({p.returncode}); last lines:\\n" + "".join(tail[-15:]))
ROOT = pathlib.Path("/content/predicate-data-pipeline")
if not (ROOT / "predx").exists():
    sh(f"git clone --quiet {REPO_URL} {ROOT}")
os.chdir(ROOT)
sh(f"git checkout --quiet {REPO_REF} && git log --oneline -1")
for d in ["data", "fdasearch/results"]: (ROOT / d).mkdir(parents=True, exist_ok=True)
import torch; print("python", sys.version.split()[0], "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
"""))

cells.append(md("""
## 2. Data
**drive**: the bundle is copied from Drive (mounted, or fetched by folder id), every file is checked against the
sha256 in `manifest.json`, and `data/dataset/` is the predx build as shipped (or rebuilt from the parsed records when
`MASK_LEVEL` is set). In `quick` mode the baselines restrict the corpus and the queries to one panel.

**fda**: `quick` runs `predx pilot` (metadata, a stratified sample of subjects from one panel, fetch, text with the
GPU backend, parse, the predicates they cite, build, audit). `full` runs `predx run-all`. The fetch stage sends an
identifying User-Agent (set `CONTACT_EMAIL` if you want your address in it) and rate-limits itself.
"""))
cells.append(code("""
import hashlib, shutil
PREDX = f"python -m predx.cli --config config/pymupdf.yaml"
panel_arg = f"--panel {PANEL}" if MODE == "quick" else ""

def fetch_bundle():
    src = pathlib.Path(DATA_DIR)
    if not src.exists():
        try:
            from google.colab import drive; drive.mount("/content/drive")
        except Exception as e:
            print("Drive mount not available:", e)
    if not src.exists() and DATA_FOLDER_ID:
        sh("pip install -q gdown")
        src = pathlib.Path("/content/fdasearch_data_v3")
        sh(f"gdown --folder https://drive.google.com/drive/folders/{DATA_FOLDER_ID} -O {src}")
    assert (src / "manifest.json").exists(), f"bundle not found at {src}: mount Drive with fdasearch_data_v3 in My Drive, or set DATA_FOLDER_ID"
    man = json.load(open(src / "manifest.json"))
    for f in man["files"]:
        rel = f["path"]      # paths inside the bundle mirror the predx data root; reference_results/ goes to fdasearch/results/reference/
        dst = ROOT / ("fdasearch/results/reference/" + rel.split("/", 1)[1] if rel.startswith("reference_results/") else "data/" + rel)
        if dst.exists() and dst.stat().st_size == f["bytes"]:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"  {rel} ({f['bytes']/1e6:.0f} MB)", flush=True)
        shutil.copyfile(src / rel, dst)
        h = hashlib.sha256()
        with open(dst, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""): h.update(chunk)
        assert h.hexdigest() == f["sha256"], f"checksum mismatch for {rel}"
    if MASK_LEVEL:
        sh(f"{PREDX} build --mask-level {MASK_LEVEL}")
        sh(f"{PREDX} validate")
    print("bundle ready")

if DATA_SOURCE == "drive":
    fetch_bundle()
elif MODE == "quick":
    sh(f"{PREDX} pilot --panel {PANEL} --n {PILOT_N} --no-tokenizers")
    panel_arg = ""     # the pilot build is already one panel
else:
    sh(f"{PREDX} run-all --no-tokenizers")
card = json.load(open("data/dataset/dataset_card.json"))
print("predx build:", card["built_at"], "| mask_level", card["mask_level"], "| corpus", card["n_corpus"], "| queries", {k: v for k, v in card["counts"].items() if k.endswith("retrievable")})
"""))

cells.append(md("""
## 3. Baselines
Candidates for each query are the corpus documents cleared before the subject (`n_candidates` in the predx build);
gold = every cited predicate that is in the corpus. `--pc` additionally scores each method restricted to the
subject's product code. Embeddings are cached per K-number in `fdasearch/results/emb/`.
"""))
cells.append(code("""
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN            # read by huggingface_hub in the subprocesses below
    from huggingface_hub import login; login(token=HF_TOKEN)
import transformers; print("transformers", transformers.__version__, "(EmbeddingGemma needs >= 4.56)")
if GEMINI_API_KEY:
    open("fdasearch/.env", "w").write(f"GEMINI_API_KEY={GEMINI_API_KEY}\\n")
for stale in ["fdasearch/results/metrics.csv", "fdasearch/results/REPORT.md"]:   # the repo ships the full-run numbers; start this run clean
    if os.path.exists(stale): os.remove(stale)
BASE = f"python fdasearch/scripts/run_baselines.py --dataset data/dataset {panel_arg}"

sh(f"{BASE} --pc --methods random,pop,bm25,tfidf,lsa")
dense_specs = ["emb:" + m for m in EMBED_MODELS if HF_TOKEN or "gemma" not in m]
if RUN_GEMINI and GEMINI_API_KEY:
    dense_specs.append("gemini:gemini-embedding-001")
if dense_specs:
    sh(f"{BASE} --pc --methods " + ",".join(dense_specs))
"""))
cells.append(md("""
### 3b. Hybrids
Reciprocal-rank fusion of the saved top-100 lists (`rrf:a|b`; names are the sanitized method names in `fdasearch/results/top/`).
"""))
cells.append(code("""
import re
tags = [re.sub(r"[^A-Za-z0-9_.+-]", "_", m.replace("/", "__")) for m in dense_specs]      # e.g. emb_google__embeddinggemma-300m, gemini_gemini-embedding-001
fusions = [f"rrf:tfidf|{t}" for t in tags] + [f"rrf:bm25|{t}" for t in tags[:1]]
if fusions:
    sh(f'{BASE} --methods "' + ",".join(fusions) + '"')
"""))

cells.append(md("## 4. Report"))
cells.append(code("""
sh(f"python fdasearch/scripts/summarize_results.py --dataset data/dataset {panel_arg} > /dev/null")
from IPython.display import Markdown, display
display(Markdown(open("fdasearch/results/REPORT.md").read()))
"""))
cells.append(md("""
### Reproducibility check
With `DATA_SOURCE="drive"`, `MODE="full"` and `MASK_LEVEL=None` the lexical methods must match the reference run to
the fourth decimal (same texts, same code); dense models re-embedded here should land within a few tenths of a point
(GPU/batch nondeterminism). The reference EmbeddingGemma run went through Ollama (`ollama:embeddinggemma`), so it
has no direct counterpart. In `quick` mode the pool is one panel, so numbers differ by design.
"""))
cells.append(code("""
import pandas as pd
ref = "fdasearch/results/reference/metrics.csv"
if os.path.exists(ref):
    a = pd.read_csv("fdasearch/results/metrics.csv").drop_duplicates(["method", "split"], keep="last").set_index(["method", "split"])
    b = pd.read_csv(ref).drop_duplicates(["method", "split"], keep="last").set_index(["method", "split"])
    common = a.index.intersection(b.index)
    cmp = pd.DataFrame({"this_run": a.loc[common, "recall10"], "reference": b.loc[common, "recall10"]})
    cmp["diff"] = (cmp.this_run - cmp.reference).round(4)
    print(f"{len(common)} (method, split) pairs in common; max |diff| in Recall@10 = {cmp['diff'].abs().max():.4f}")
    display(cmp.sort_values("reference", ascending=False).head(20))
"""))
cells.append(md("""
### Look at one query
What the model sees (K-numbers and predicate names masked by `predx build`), the cited predicates, and the
TF-IDF top 10.
"""))
cells.append(code("""
import numpy as np
sys.path.insert(0, "fdasearch/scripts")
from predx_dataset import load_dataset
c, q, pairs, card = load_dataset("data/dataset", panel=PANEL if panel_arg else None)
top = np.load("fdasearch/results/top/tfidf.npy")           # rows follow the dev+test query order used by run_baselines
i = int(q.index[q.split == "test"][0])                      # first test query; change i to browse
r = q.iloc[i]
print(r.k, r.date, r.product_code, "|", r.panel, "| candidates:", r.n_candidates); print(r.query[:600], "...\\n")
print("gold in corpus:", r.positives, "| primary:", r.primary, "| cited in total:", r.n_gold)
for rank, j in enumerate(top[i][:10], 1):
    d = c.iloc[j]; mark = "<-- predicate" if d.k in set(r.positives) else ""
    print(f"{rank:2d}. {d.k} {d.date} {d.product_code} {d.device_name[:60]} {mark}")
"""))

cells.append(md("""
## 5. Keep the results
Zips the metrics, report, per-query ranks and the dataset card. Mount Drive first if you want them to survive the runtime.
"""))
cells.append(code("""
import zipfile
with zipfile.ZipFile("/content/fdasearch_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in ["fdasearch/results/metrics.csv", "fdasearch/results/REPORT.md", "data/dataset/dataset_card.json", "data/reports/audit.md"]:
        if os.path.exists(p): z.write(p)
    for f in glob.glob("fdasearch/results/ranks/*"): z.write(f)
print("/content/fdasearch_results.zip", f"{os.path.getsize('/content/fdasearch_results.zip')/1e6:.1f} MB")
try:
    from google.colab import files; files.download("/content/fdasearch_results.zip")
except Exception as e:
    print("not in Colab:", e)
# from google.colab import drive; drive.mount('/content/drive'); shutil.copy('/content/fdasearch_results.zip', '/content/drive/MyDrive/')
"""))

nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
    "colab": {"name": "fdasearch_510k_baselines.ipynb", "provenance": [], "toc_visible": True},
    "accelerator": "GPU"}, "cells": cells}
for i, cl in enumerate(cells):
    cl["id"] = f"cell-{i}"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote", OUT, len(cells), "cells", f"{os.path.getsize(OUT)/1e3:.0f} KB")
