#!/usr/bin/env bash
# Build the Drive bundle fdasearch_data_v3 (a predx build + what is needed to re-run `predx build`) in E:/fdasearch/bundle_v3.
# Usage: bash bundle_build.sh [PREDX_DATA_ROOT]      (default E:/predx_data)
set -e
SRC=${1:-E:/predx_data}
OUT=/e/fdasearch/bundle_v3
mkdir -p "$OUT/dataset" "$OUT/reports" "$OUT/reference_results"
cp "$SRC"/dataset/*.jsonl.gz "$SRC"/dataset/dataset_card.json "$OUT/dataset/"
cp "$SRC"/parsed.jsonl.gz "$SRC"/text_stats.jsonl "$SRC"/fetch_status.jsonl "$OUT/"
mkdir -p "$OUT/meta" && cp "$SRC"/meta/510k.jsonl.gz "$OUT/meta/"
cp "$SRC"/reports/audit.md "$SRC"/reports/audit.json "$OUT/reports/" 2>/dev/null || true
cp /e/fdasearch/results/metrics.csv /e/fdasearch/results/REPORT.md "$OUT/reference_results/"
/e/fdasearch/.venv/Scripts/python.exe -X utf8 - "$OUT" <<'PY'
import os, sys, json, hashlib, datetime
root = sys.argv[1]; m = []
for d, _, fs in os.walk(root):
    for f in fs:
        if f == "manifest.json":
            continue
        p = os.path.join(d, f); rel = os.path.relpath(p, root).replace(os.sep, "/")
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        m.append(dict(path=rel, bytes=os.path.getsize(p), sha256=h.hexdigest()))
m.sort(key=lambda x: x["path"])
card = json.load(open(os.path.join(root, "dataset", "dataset_card.json")))
json.dump(dict(built=datetime.date.today().isoformat(), dataset="predx build " + card["built_at"] + " (mask_level " + card["mask_level"] + ")",
               layout="paths mirror the predx data root (paths.data_dir); reference_results/ are fdasearch's metrics on this build", files=m),
          open(os.path.join(root, "manifest.json"), "w"), indent=1)
print(f"{sum(x['bytes'] for x in m) / 1e6:.0f} MB, {len(m)} files")
for x in m:
    print(f"{x['bytes'] / 1e6:8.1f} MB  {x['path']}")
PY
