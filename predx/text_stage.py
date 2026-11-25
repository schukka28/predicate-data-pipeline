"""Stage `text`: PDFs -> cached per-page text (data/text/yy/K.txt.gz) + text_stats.jsonl.

The expensive step (OCR) is cached, so the parsing rules can be iterated on without re-running OCR.
Resumable: documents whose cache file exists are skipped.
"""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor

from .io import append_jsonl, save_pages

log = logging.getLogger("predx.text")


def extract_pdf(pdf, tcfg: dict) -> dict:
    """Dispatch on text.backend: pdftotext (poppler + tesseract, default) or pymupdf (PyMuPDF + RapidOCR on GPU)."""
    backend = tcfg.get("backend", "pdftotext")
    if backend == "pymupdf":
        from .pdftext_pymupdf import extract_pdf as ex
    elif backend == "pdftotext":
        from .pdftext import extract_pdf as ex
    else:
        raise ValueError(f"unknown text.backend {backend!r} (pdftotext | pymupdf)")
    return ex(pdf, tcfg)


def _work(args):
    k, pdf, tpath, tcfg = args
    backend = tcfg.get("backend", "pdftotext")
    try:
        res = extract_pdf(pdf, tcfg)
        save_pages(tpath, res.pop("pages"))
        conf = res.pop("ocr_conf", [])
        return {"k": k, **res, "ocr_pages": len(res["ocr_pages"]), "ocr_conf_mean": round(sum(conf) / len(conf), 1) if conf else None,
                "ocr_conf_min": min(conf) if conf else None, "backend": backend, "error": None}
    except Exception as e:  # keep going; the audit reports these as text_failed
        return {"k": k, "n_pages": 0, "n_pages_read": 0, "ocr_pages": 0, "n_chars": 0, "seconds": 0.0, "backend": backend, "error": repr(e)[:300]}


def run_text(cfg: dict, paths, ks, force: bool = False) -> dict:
    tcfg = cfg["text"]
    todo, missing_pdf = [], 0
    for k in ks:
        if not force and paths.text(k).exists():
            continue
        if not paths.pdf(k).exists():
            missing_pdf += 1
            continue
        todo.append((k, str(paths.pdf(k)), str(paths.text(k)), tcfg))
    workers = tcfg.get("workers") or os.cpu_count() or 1
    log.info("text: %d to extract (%d skipped: no PDF), %d workers", len(todo), missing_pdf, workers)
    ok = err = ocr_docs = 0
    t0 = time.time()
    if not todo:
        return {"extracted": 0, "errors": 0, "no_pdf": missing_pdf}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, st in enumerate(ex.map(_work, todo, chunksize=2), 1):
            append_jsonl(paths.text_stats, st)
            if st["error"]:
                err += 1
            else:
                ok += 1
                ocr_docs += 1 if st["ocr_pages"] else 0
                if not cfg["fetch"].get("keep_pdfs", True):
                    try:
                        paths.pdf(st["k"]).unlink()
                    except OSError:
                        pass
            if i % 200 == 0:
                el = time.time() - t0
                log.info("text: %d/%d  (%.1f docs/s, %d errors)", i, len(todo), i / el, err)
    return {"extracted": ok, "errors": err, "ocr_docs": ocr_docs, "no_pdf": missing_pdf}
