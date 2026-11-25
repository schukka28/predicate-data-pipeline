"""Label-leakage controls. Summaries often name the predicate in the very sections used as model input.

mask levels
  none     : nothing masked (for measuring leakage only)
  standard : every K/BK/DEN/P number -> [K-NUMBER]; names of the subject's cited predicates -> [DEVICE]
             (multi-word names only; skipped when the name is a part of the subject's own name)
  strict   : standard + every distinctive token of a cited predicate's name + drops sentences that talk about
             predicates / substantial equivalence. Use it for the "does performance collapse?" check.
"""
from __future__ import annotations

import re

from .knumbers import find_knumbers

K_TOKEN = "[K-NUMBER]"
NAME_TOKEN = "[DEVICE]"

GENERIC = set(
    """system device devices catheter monitor sensor software kit set assembly instrument unit probe system
    systems stent valve lead patch pump needle syringe glove gloves mask laser imaging image images digital
    portable disposable sterile single use reusable adult pediatric patient patients surgical medical
    electrode electrodes cuff tube tubes wire wires implant implants plate screw screws coil
    analyzer analyser reagent assay test tests""".split()
)
_PRED_SENT = re.compile(r"predicate|substantial(?:ly)?\s+equivalen|equivalent\s+to", re.I)


def mask_k_numbers(text: str, repl: str = K_TOKEN):
    n = 0
    while True:  # a second pass catches numbers glued to a previous one ("K140833K140833"), which the lookbehind skips
        hits = find_knumbers(text, known=None, accept_unverified_ocr=True)
        if not hits:
            return text, n
        for h in sorted(hits, key=lambda h: -h.start):
            text = text[: h.start] + repl + text[h.end:]
        n += len(hits)


def _norm_name(n: str) -> str:
    n = re.sub(r"[™®©]", "", n or "")
    n = re.sub(r"\s+", " ", n)
    return n.strip().lower()


def _name_regex(name: str):
    toks = [re.escape(t) for t in re.split(r"\s+", name.strip()) if t]
    return re.compile(r"(?<![A-Za-z0-9])" + r"\s+".join(toks) + r"(?![A-Za-z0-9])", re.I)


def mask_names(text: str, names, own_name: str = "", min_chars: int = 8, min_tokens: int = 2):
    own = _norm_name(own_name)
    n_masked = 0
    for name in sorted({_norm_name(n) for n in names if n}, key=len, reverse=True):
        if len(name) < min_chars or len(name.split()) < min_tokens:
            continue
        if own and name in own:
            continue
        text, n = _name_regex(name).subn(NAME_TOKEN, text)
        n_masked += n
    return text, n_masked


def distinctive_tokens(name: str, own_name: str = "") -> set[str]:
    own = set(re.findall(r"[A-Za-z0-9]+", _norm_name(own_name)))
    out = set()
    for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", re.sub(r"[™®©]", "", name or "")):
        low = tok.lower()
        if len(low) < 4 or low in GENERIC or low in own:
            continue
        if any(c.isdigit() for c in tok) or tok[0].isupper():
            out.add(low)
    return out


def drop_predicate_sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+", text)
    keep = [p for p in parts if not _PRED_SENT.search(p)]
    return " ".join(keep), len(parts) - len(keep)


def mask_doc_text(text: str, level: str):
    """Masking applied to every corpus document (queries and candidates alike)."""
    if not text or level == "none":
        return text, {"k": 0, "names": 0, "sentences": 0}
    text, nk = mask_k_numbers(text)
    ns = 0
    if level == "strict":
        text, ns = drop_predicate_sentences(text)
    return text, {"k": nk, "names": 0, "sentences": ns}


def mask_query_text(text: str, level: str, gold_names, own_name: str = "", cfg: dict | None = None):
    """doc masking + masking of the subject's cited predicates' names (queries only)."""
    lc = (cfg or {}).get("leakage", {})
    text, st = mask_doc_text(text, level)
    if not text or level == "none":
        return text, st
    text, nn = mask_names(text, gold_names, own_name, lc.get("name_min_chars", 8), lc.get("name_min_tokens", 2))
    st["names"] = nn
    if level == "strict":
        toks = set()
        for n in gold_names:
            toks |= distinctive_tokens(n, own_name)
        for t in sorted(toks, key=len, reverse=True):
            text, c = re.subn(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![A-Za-z0-9])", NAME_TOKEN, text, flags=re.I)
            st["names"] += c
    return text, st


def leak_stats(raw_ifu: str, raw_desc: str, gold_ks, gold_names, own_k: str) -> dict:
    """Leakage present in the RAW (pre-masking) query text."""
    text = " ".join(x for x in (raw_ifu, raw_desc) if x)
    hits = [h for h in find_knumbers(text, known=None, accept_unverified_ocr=True) if h.k != own_k]
    gold = set(gold_ks)
    name_hits = 0
    own_name = ""
    for n in gold_names:
        nn = _norm_name(n)
        if len(nn) >= 8 and len(nn.split()) >= 2:
            name_hits += len(_name_regex(nn).findall(text))
    return {
        "raw_other_k": len(hits),
        "raw_gold_k": sum(1 for h in hits if h.k in gold),
        "pred_language": bool(_PRED_SENT.search(text)),
        "raw_gold_name": name_hits,
    }


def residual_leak(text: str, gold_names, own_name: str = "") -> dict:
    hits = find_knumbers(text, known=None, accept_unverified_ocr=True)
    own = _norm_name(own_name)
    nh = 0
    for n in gold_names:
        nn = _norm_name(n)
        if len(nn) >= 8 and len(nn.split()) >= 2 and not (own and nn in own):
            nh += len(_name_regex(nn).findall(text))
    return {"residual_k": len(hits), "residual_gold_name": nh}
