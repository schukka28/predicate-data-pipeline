"""Put PDFs that fdasearch already downloaded into a predx data root, so the predx stages (text, parse, build,
validate, audit) run on them without re-downloading anything.

Usage:
  python scripts/predx_bridge.py --predx-root E:/predx_data [--predx-repo ../predicate-data-pipeline]
         [--pdf-dir data/pdfs --pdf-dir data/pdfs_closure] [--openfda data/raw/device-510k.json.zip]

Writes, under --predx-root (predx's paths.data_dir):
  raw/device-510k.json.zip      copy of the openFDA bulk file
  meta/510k.jsonl.gz            normalised metadata (predx.metadata.normalise_record), same as `predx meta`
  pdf/<yy>/K.pdf                hard links to the downloaded PDFs (falls back to copying across drives)
  fetch_status.jsonl            one "ok" row per PDF present (predx's fetch stage was not run)
Then:
  python -m predx.cli --config config/pymupdf.yaml --set paths.data_dir=E:/predx_data text --queue all
  python -m predx.cli --set paths.data_dir=E:/predx_data parse
  python -m predx.cli --set paths.data_dir=E:/predx_data build
  python -m predx.cli --set paths.data_dir=E:/predx_data validate
  python -m predx.cli --set paths.data_dir=E:/predx_data audit
"""
import argparse, glob, json, os, shutil, sys, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predx-root", required=True, help="predx paths.data_dir to populate")
    ap.add_argument("--predx-repo", default=None, help="checkout of predicate-data-pipeline (default: parent of this fdasearch/ folder, else ../predicate-data-pipeline)")
    ap.add_argument("--pdf-dir", action="append", default=None, help="folder(s) of K*.pdf (default: data/pdfs and data/pdfs_closure)")
    ap.add_argument("--openfda", default=os.path.join(ROOT, "data", "raw", "device-510k.json.zip"))
    a = ap.parse_args()

    for cand in [a.predx_repo, os.path.dirname(ROOT), os.path.join(os.path.dirname(ROOT), "predicate-data-pipeline")]:
        if cand and os.path.isdir(os.path.join(cand, "predx")):
            sys.path.insert(0, os.path.abspath(cand))
            break
    from predx.config import Paths, load_config, shard
    from predx.io import append_jsonl, write_jsonl
    from predx.metadata import normalise_record

    paths = Paths(load_config(), a.predx_root)
    paths.root.mkdir(parents=True, exist_ok=True)

    # metadata from the local openFDA bulk file (same normalisation as `predx meta`)
    if paths.meta.exists():
        print("meta already present:", paths.meta)
    else:
        paths.raw.mkdir(parents=True, exist_ok=True)
        dst = paths.raw / os.path.basename(a.openfda)
        if not dst.exists():
            shutil.copyfile(a.openfda, dst)
        rows, seen = [], set()
        with zipfile.ZipFile(a.openfda) as z:
            for name in z.namelist():
                if name.endswith(".json"):
                    for rec in json.loads(z.read(name)).get("results", []):
                        n = normalise_record(rec)
                        if n and n["k"] not in seen:
                            seen.add(n["k"]); rows.append(n)
        rows.sort(key=lambda x: (x["decision_date"] or "", x["k"]))
        write_jsonl(paths.meta, rows)
        print("meta:", len(rows), "records ->", paths.meta)

    # PDFs: hard links into pdf/<yy>/K.pdf
    dirs = a.pdf_dir or [os.path.join(ROOT, "data", "pdfs"), os.path.join(ROOT, "data", "pdfs_closure")]
    linked = present = 0
    status_rows = []
    for d in dirs:
        for src in sorted(glob.glob(os.path.join(d, "*.pdf"))):
            k = os.path.basename(src)[:-4].upper()
            dst = paths.pdf(k)
            if dst.exists():
                present += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copyfile(src, dst)
            linked += 1
            status_rows.append({"k": k, "status": "ok", "url": None, "folder": None, "bytes": os.path.getsize(src), "error": None,
                                "source": "fdasearch"})
    for r in status_rows:
        append_jsonl(paths.fetch_status, r)
    print(f"pdfs: {linked} linked, {present} already present -> {paths.pdf_dir}")
    print("\nnext:\n  python -m predx.cli --config config/pymupdf.yaml --set paths.data_dir=%s text --queue all" % a.predx_root)


if __name__ == "__main__":
    main()
