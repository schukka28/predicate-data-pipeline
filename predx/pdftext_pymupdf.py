"""PDF -> per-page text with PyMuPDF; image-only or garbled pages are OCR'd with RapidOCR (ONNX Runtime, CUDA when
available). A drop-in alternative to pdftext.extract_pdf with the same return contract, selected with

    text:
      backend: pymupdf          # default: pdftotext (poppler + tesseract)

Needs no poppler or tesseract binaries: pip install pymupdf rapidocr-onnxruntime onnxruntime-gpu  (CUDA 12 build; the
CPU onnxruntime also works, ~10x slower). Measured on an RTX 4090 at 150 dpi: ~0.65 s per OCR'd page; the 510(k)
corpus has ~1,000 image-only summaries out of ~49,500, so the whole text stage is well under an hour on one GPU.

Recognizer: RapidOCR ships the Chinese+English PP-OCRv4 recognizer, which drops the spaces between English words
("PremarketNotificationNumber:K173428"), and the heading rules need those spaces. `ocr.rec_model` names an English
recognizer on the Hugging Face hub as "<repo>:<file>" (default in config/pymupdf.yaml: SWHL/RapidOCR's
PP-OCRv3/en_PP-OCRv3_rec_infer.onnx, 9 MB, downloaded once into the HF cache); it keeps the spaces and reads the
same pages with a higher score. If the download fails the bundled recognizer is used and a warning is logged.

Differences from the pdftotext backend: no K-number digit re-read vote (k_reread_changed is always 0), and
ocr_conf_mean is RapidOCR's mean line recognition score scaled to 0-100 rather than tesseract's word confidence.
"""
from __future__ import annotations

import logging
import time

from .pdftext import _alnum, looks_garbled

log = logging.getLogger("predx.text")
_OCR = None


def _rec_model_path(spec: str | None):
    if not spec:
        return None
    try:
        from huggingface_hub import hf_hub_download

        repo, fname = spec.split(":", 1)
        return hf_hub_download(repo, fname)
    except Exception as e:  # offline, or the hub is down: fall back to the bundled recognizer
        log.warning("ocr.rec_model %s not available (%s); using RapidOCR's bundled recognizer", spec, repr(e)[:120])
        return None


def _ocr(ocfg: dict):
    """One RapidOCR instance per process (the text stage runs in a process pool)."""
    global _OCR
    if _OCR is None:
        try:
            import torch  # noqa: F401  loads the CUDA runtime DLLs that onnxruntime-gpu links against (Windows)
        except ImportError:
            pass
        import onnxruntime
        from rapidocr_onnxruntime import RapidOCR

        cuda = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
        kw = {}
        rec = _rec_model_path(ocfg.get("rec_model"))
        if rec:
            kw["rec_model_path"] = rec
        _OCR = RapidOCR(det_use_cuda=cuda, rec_use_cuda=cuda, cls_use_cuda=cuda, **kw)
    return _OCR


def ocr_page_conf(page, dpi: int, ocfg: dict):
    """-> (text, mean recognition score 0-100 or None). Lines are ordered top-to-bottom then left-to-right so section
    headings stay on their own line, which is what sections.find_headings relies on."""
    import numpy as np
    import pymupdf

    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    res, _ = _ocr(ocfg)(img)
    if not res:
        return "", None
    res = sorted(res, key=lambda r: (round(r[0][0][1] / 12), r[0][0][0]))
    text = "\n".join(r[1] for r in res) + "\n"
    conf = 100.0 * sum(float(r[2]) for r in res) / len(res)
    return text, conf


def extract_pdf(path, tcfg: dict) -> dict:
    import pymupdf

    t0 = time.time()
    doc = pymupdf.open(str(path))
    n = doc.page_count
    if n <= 0:
        raise ValueError("unreadable PDF (0 pages)")
    n_read = min(n, tcfg["max_pages"])
    pages = [doc[i].get_text().replace("\f", "\n") for i in range(n_read)]
    ocr_pages: list[int] = []
    ocr_conf: list[float] = []
    ocfg = tcfg["ocr"]
    if ocfg["enabled"]:
        for i, t in enumerate(pages):
            if len(ocr_pages) >= ocfg["max_pages"]:
                break
            if _alnum(t) < ocfg["min_alnum_chars"] or looks_garbled(t):
                txt, conf = ocr_page_conf(doc[i], ocfg["dpi"], ocfg)
                if _alnum(txt) > _alnum(t):
                    pages[i] = txt
                    ocr_pages.append(i)
                    if conf is not None:
                        ocr_conf.append(round(conf, 1))
    doc.close()
    return {
        "pages": pages,
        "n_pages": n,
        "n_pages_read": len(pages),
        "ocr_pages": ocr_pages,
        "k_reread_changed": 0,
        "ocr_conf": ocr_conf,
        "n_chars": sum(len(p) for p in pages),
        "seconds": round(time.time() - t0, 3),
    }
