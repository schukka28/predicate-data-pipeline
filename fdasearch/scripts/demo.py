"""Local demo: type a device's Indications for Use and Device Description, get ranked candidate predicates.

  python scripts/demo.py --dataset E:/predx_data/dataset
  then open http://127.0.0.1:8510

Search is exact (one matrix product over the 52k corpus vectors); the default method is reciprocal-rank fusion of the
fine-tuned embedder and TF-IDF, the best system in results/REPORT.md. Document vectors come from the cache written by
run_baselines.py (results/emb/<tag>.d.npz); missing ones are embedded on first start. Standard library server, bound to
localhost only. "Load a test example" fills the form with a held-out 2025-26 subject and marks its cited predicates.
"""
import argparse, json, os, pickle, random, re, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predx_dataset import load_dataset, input_text
from run_baselines import prompts_for, cached_embed, embed_st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNUM = re.compile(r"(?<![A-Za-z0-9])(?:K|BK|DEN|P)\d{6}(?![0-9])")
FDA_URL = "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID="

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>510(k) predicate search</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1a1f29;--mute:#5d6778;--line:#d9dee7;--acc:#1f5fbf;--hit:#e6f4ea;--hitink:#1b6b36}
@media (prefers-color-scheme: dark){:root{--bg:#12151b;--card:#1b2029;--ink:#e8ebf1;--mute:#9aa4b5;--line:#2e3542;--acc:#7fb0ff;--hit:#173524;--hitink:#8fd8a8}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:20px 16px 60px}h1{font-size:22px;margin:0 0 4px}p.sub{color:var(--mute);margin:0 0 16px}
label{display:block;font-weight:600;margin:12px 0 4px}textarea,select,input{width:100%;font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px}
textarea{min-height:110px;resize:vertical}.row{display:flex;gap:12px;flex-wrap:wrap}.row>div{flex:1 1 150px}
button{font:inherit;font-weight:600;padding:9px 16px;border-radius:6px;border:1px solid var(--acc);background:var(--acc);color:#fff;cursor:pointer;margin:16px 8px 0 0}
button.alt{background:transparent;color:var(--acc)}button:disabled{opacity:.5}
#status{color:var(--mute);margin:14px 0 8px;min-height:1.4em}
.res{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin:8px 0}
.res.gold{background:var(--hit);border-color:var(--hitink)}.res h3{margin:0;font-size:16px}.res a{color:var(--acc)}
.meta{color:var(--mute);font-size:13px;margin:2px 0 6px}.snip{font-size:14px;overflow-wrap:anywhere}
.tag{display:inline-block;font-size:12px;font-weight:700;color:var(--hitink);margin-left:8px}
details{margin-top:6px}summary{cursor:pointer;color:var(--acc);font-size:13px}
</style></head><body><main>
<h1>510(k) predicate search</h1>
<p class="sub">Describe a new device. The tool ranks earlier 510(k) clearances that it could cite as predicates. Corpus: __NDOCS__ summaries, __DMIN__ to __DMAX__. Research demo, not regulatory advice.</p>
<label for="ifu">Indications for Use</label><textarea id="ifu"></textarea>
<label for="desc">Device Description</label><textarea id="desc"></textarea>
<div class="row">
<div><label for="method">Method</label><select id="method"><option value="hybrid">Fine-tuned embedder + TF-IDF</option><option value="dense">Fine-tuned embedder</option><option value="tfidf">TF-IDF</option></select></div>
<div><label for="pc">Product code (optional)</label><input id="pc" maxlength="3" placeholder="e.g. DQY"></div>
<div><label for="before">Cleared before (optional)</label><input id="before" type="date"></div>
<div><label for="topk">Results</label><select id="topk"><option>10</option><option>25</option><option>50</option></select></div>
</div>
<button id="go">Search</button><button id="ex" class="alt">Load a test example</button>
<div id="status"></div><div id="out"></div>
</main><script>
const $ = id => document.getElementById(id);
let gold = [];
function esc(s){const d=document.createElement('div');d.textContent=s==null?'':String(s);return d.innerHTML}
async function search(){
  $('go').disabled=true;$('status').textContent='Searching...';$('out').innerHTML='';
  try{
    const r=await fetch('/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ifu:$('ifu').value,desc:$('desc').value,method:$('method').value,pc:$('pc').value,before:$('before').value,topk:+$('topk').value,gold:gold})});
    const j=await r.json();
    if(j.error){$('status').textContent=j.error;return}
    let s=j.n_candidates.toLocaleString()+' candidates searched in '+j.ms+' ms.';
    if(gold.length){s+=' Cited predicates in the corpus: '+j.gold_ranks.map(g=>g.k+' (rank '+(g.rank||'not in the ranked list')+')').join(', ')+'.'}
    $('status').textContent=s;
    $('out').innerHTML=j.results.map((d,i)=>'<div class="res'+(d.gold?' gold':'')+'"><h3>'+(i+1)+'. <a target="_blank" rel="noopener" href="'+esc(d.url)+'">'+esc(d.k)+'</a> '+esc(d.device_name)+(d.gold?'<span class="tag">CITED PREDICATE</span>':'')+'</h3><div class="meta">'+esc(d.applicant)+' · cleared '+esc(d.date)+' · product code '+esc(d.product_code)+' · panel '+esc(d.panel)+' · score '+d.score+'</div><div class="snip">'+esc(d.ifu)+'</div><details><summary>Device description</summary><div class="snip">'+esc(d.description)+'</div></details></div>').join('');
  }catch(e){$('status').textContent='Request failed: '+e}
  finally{$('go').disabled=false}
}
$('go').onclick=()=>search();
['ifu','desc'].forEach(id=>$(id).addEventListener('input',()=>{gold=[]}));
$('ex').onclick=async()=>{
  const j=await (await fetch('/example')).json();
  $('ifu').value=j.ifu;$('desc').value=j.description;$('before').value=j.date;$('pc').value='';gold=j.gold;
  $('status').textContent='Loaded '+j.k+' ('+j.device_name+', cleared '+j.date+'). Its summary cites: '+j.gold.join(', ')+'.';$('out').innerHTML='';
};
</script></body></html>"""


class Engine:
    def __init__(self, a):
        t0 = time.time()
        self.corpus, self.queries, _, self.card = load_dataset(a.dataset, splits=("test",))
        c = self.corpus
        self.kidx = {k: i for i, k in enumerate(c.k)}
        self.dates = c.date.values
        self.pcs = c.product_code.values
        self.tpl = self.card["config"].get("input_template") or {"ifu_header": "", "desc_header": "", "separator": " "}
        d = a.dataset or os.environ.get("PREDX_DATASET") or os.path.join(ROOT, "data", "dataset")
        self.examples = {}
        import gzip
        with gzip.open(os.path.join(d, "queries_test.jsonl.gz"), "rt", encoding="utf-8") as f:
            for line in f:
                q = json.loads(line)
                if q.get("gold_in_corpus"):
                    self.examples[q["k"]] = q
        self.example_keys = sorted(self.examples)
        print(f"corpus {len(c)} loaded ({time.time() - t0:.0f}s)", flush=True)

        # TF-IDF, same settings as run_baselines; cached next to the model because fitting takes about a minute
        cache = os.path.join(ROOT, "models", f"demo_tfidf_{self.card['built_at'][:10]}_{len(c)}.pkl")
        if os.path.exists(cache):
            self.tv, self.DT = pickle.load(open(cache, "rb"))
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tv = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.5, stop_words="english", ngram_range=(1, 2))
            self.DT = self.tv.fit_transform(c.text.tolist()).T.tocsr()
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            pickle.dump((self.tv, self.DT), open(cache, "wb"))
        print(f"tf-idf ready ({time.time() - t0:.0f}s)", flush=True)

        import torch
        from sentence_transformers import SentenceTransformer
        tag = a.model.replace("/", "__")  # same cache name as run_baselines
        self.D = cached_embed(tag, lambda t, isq: embed_st(a.model, t, isq, max_len=a.max_len), c.text.tolist(), c.k.tolist(), False)
        dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        kw = {"trust_remote_code": True}
        if dev == "cuda" and torch.cuda.is_bf16_supported():
            kw["model_kwargs"] = {"torch_dtype": torch.bfloat16}
        self.model = SentenceTransformer(a.model, device=dev, **kw)
        self.model.max_seq_length = a.max_len
        self.qp = prompts_for(a.model)[0]
        print(f"embedder on {dev}, {self.D.shape[0]} document vectors ready ({time.time() - t0:.0f}s)", flush=True)

    def search(self, r):
        t0 = time.time()
        ifu, desc = (r.get("ifu") or "").strip(), (r.get("desc") or "").strip()
        if not ifu and not desc:
            return {"error": "Enter the Indications for Use, the Device Description, or both."}
        text = KNUM.sub("[K-NUMBER]", input_text(ifu, desc, self.tpl))  # corpus text has K-numbers masked the same way
        mask = np.ones(len(self.corpus), dtype=bool)
        if r.get("before"):
            mask &= self.dates < str(r["before"])
        pc = (r.get("pc") or "").strip().upper()
        if pc:
            mask &= self.pcs == pc
        if not mask.any():
            return {"error": "No corpus device matches the product code and date filter."}
        method = r.get("method") or "hybrid"
        lists = []
        if method in ("hybrid", "dense"):
            q = self.model.encode([self.qp + text], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)[0]
            lists.append(self.D @ q)
        if method in ("hybrid", "tfidf"):
            lists.append((self.tv.transform([text]) @ self.DT).toarray().ravel())
        if len(lists) == 1:
            s = np.where(mask, lists[0], -np.inf)
        else:  # reciprocal-rank fusion (k = 60) of each method's top 100, as in run_baselines
            s = np.zeros(len(self.corpus))
            for x in lists:
                x = np.where(mask, x, -np.inf)
                top = np.argsort(-x, kind="stable")[:100]
                top = top[np.isfinite(x[top])]
                s[top] += 1.0 / (60 + np.arange(1, len(top) + 1))
            s = np.where(mask & (s > 0), s, -np.inf)
        order = np.argsort(-s, kind="stable")
        order = order[np.isfinite(s[order])]
        gold = [g for g in (r.get("gold") or []) if g in self.kidx]
        rank = {int(i): n + 1 for n, i in enumerate(order)}
        c = self.corpus
        res = []
        for i in order[: max(1, min(int(r.get("topk") or 10), 100))]:
            row = c.iloc[int(i)]
            res.append({"k": row.k, "url": FDA_URL + row.k, "device_name": row.device_name, "applicant": row.applicant,
                        "date": row.date, "product_code": row.product_code, "panel": row.panel,
                        "score": round(float(s[i]), 4), "gold": row.k in gold,
                        "ifu": row.ifu[:600], "description": row.description[:1500]})
        return {"results": res, "n_candidates": int(mask.sum()), "ms": int(1000 * (time.time() - t0)),
                "gold_ranks": [{"k": g, "rank": rank.get(self.kidx[g])} for g in gold]}

    def example(self):
        q = self.examples[random.choice(self.example_keys)]
        return {"k": q["k"], "device_name": q.get("device_name", ""), "date": q["date"], "ifu": q["ifu"],
                "description": q["description"], "gold": q["gold_in_corpus"]}


def make_handler(eng):
    page = (PAGE.replace("__NDOCS__", f"{len(eng.corpus):,}").replace("__DMIN__", eng.corpus.date.min()[:4])
            .replace("__DMAX__", eng.corpus.date.max()[:4])).encode("utf-8")

    class H(BaseHTTPRequestHandler):
        def _send(self, body, ctype, code=200):
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self._send(page, "text/html; charset=utf-8")
            elif self.path == "/example":
                self._send(json.dumps(eng.example()).encode(), "application/json")
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            if self.path != "/search":
                return self._send(b"not found", "text/plain", 404)
            try:
                n = min(int(self.headers.get("Content-Length") or 0), 2_000_000)
                out = eng.search(json.loads(self.rfile.read(n) or b"{}"))
            except Exception as e:  # show the failure in the page instead of dropping the connection
                out = {"error": f"{type(e).__name__}: {e}"}
            self._send(json.dumps(out).encode(), "application/json")

        def log_message(self, *a):
            pass
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--model", default="models/embeddinggemma-300m-ft-eval1024/final",
                    help="the cached document vectors are named after this path, so one path = one --max-len")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda when available)")
    ap.add_argument("--port", type=int, default=8510)
    a = ap.parse_args()
    eng = Engine(a)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(eng))
    print(f"ready: http://127.0.0.1:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
