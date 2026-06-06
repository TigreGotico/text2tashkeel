"""Experiment: gate rawi's marks by a second model's mark/no-mark decision.

rawi has great per-marked-position accuracy (DER* ~3%) but over-marks, and the
over-marks are *confident* (a plain confidence threshold only goes so far). Idea:
keep rawi's predicted mark at a position ONLY IF a "gate" model (bilstm) also
decides that position should carry a mark; otherwise emit no mark. This targets
the confident-but-convention-divergent over-marking that thresholding can't.

Alignment note: rawi normalizes with NFD (أ→ا), bilstm with NFC, but both yield
one position per base letter, so we index-align with a length-check fallback (if
the two bare sequences ever differ in length, that sentence falls back to plain
rawi and is counted).

    python benchmarks/rawi_agreement_gating.py --limit 4000
"""

from __future__ import annotations

import argparse
import time
import unicodedata

import numpy as np

from text2tashkeel import Diacritizer
from text2tashkeel._models import (
    _BILSTM_C2I as BC2I,
    _BILSTM_UNK as BUNK,
    _strip as nfc_strip,
)

# ── metric (same as benchmarks/benchmark.py) ───────────────────────────────
_DIAC = {chr(c) for c in range(0x64B, 0x653)}
_STRIP = _DIAC | {"ٰ"}
_SHADDA = "ّ"


def _norm(s):
    return unicodedata.normalize("NFC", s)


def _reorder(d):
    return _SHADDA + d.replace(_SHADDA, "") if (_SHADDA in d and len(d) > 1) else d


def extract(text):
    labels, cur = [], ""
    for ch in _norm(text):
        if ch in _DIAC:
            cur += ch
        else:
            if labels:
                labels[-1] = _reorder(cur)
            labels.append("")
            cur = ""
    if labels and cur:
        labels[-1] = _reorder(cur)
    return labels


def score(pred, ref):
    p, r = extract(pred), extract(ref)
    dw = dt = mw = mt = 0
    for a, b in zip(p, r):
        dt += 1
        if a != b:
            dw += 1
        if b != "":
            mt += 1
            if a != b:
                mw += 1
    extra = max(0, len(r) - len(p))
    dt += extra
    dw += extra
    pw, rw = _norm(pred).split(), _norm(ref).split()
    ww = sum(1 for a, b in zip(pw, rw) if extract(a) != extract(b))
    return dw, dt, mw, mt, ww, len(rw)


# ── gate model: bilstm per-position "is this position marked?" ─────────────
class BilstmGate:
    def __init__(self):
        self.sess = Diacritizer("bilstm").backend.sess

    def marked_mask(self, gold: str):
        """Boolean per NFC-stripped base position: did bilstm predict a mark?"""
        bare = nfc_strip(gold)
        if not bare:
            return bare, np.zeros(0, bool)
        ids = np.array([[BC2I.get(c, BUNK) for c in bare]], dtype=np.int64)
        cls = self.sess.run(["logits"], {"input_ids": ids})[0][0].argmax(-1)
        return bare, cls != 0   # class 0 == no mark


# ── rawi, optionally confidence-filtered, optionally gated ─────────────────
class RawiGated:
    def __init__(self, gate: BilstmGate | None, conf: float = 0.0):
        b = Diacritizer("rawi").backend
        self.sess, self.i2d, self.c2i, self.unk = b.sess, b.i2d, b.c2i, b.unk
        self.norm = b._normalize
        self.gate = gate
        self.conf = conf
        self.fallbacks = 0

    def diacritize(self, gold: str) -> str:
        bare = "".join(c for c in self.norm(gold) if unicodedata.category(c) != "Mn")
        if not bare:
            return gold
        ids = np.array([[self.c2i.get(c, self.unk) for c in bare]], dtype=np.int64)
        logits = self.sess.run(["output"], {"input": ids})[0][0]
        cls = logits.argmax(-1)
        if self.conf:
            m = logits.max(-1, keepdims=True)
            e = np.exp(logits - m)
            conf = (e / e.sum(-1, keepdims=True))[np.arange(len(cls)), cls]
        gate_mask = None
        if self.gate is not None:
            _, gmask = self.gate.marked_mask(gold)
            if len(gmask) == len(bare):
                gate_mask = gmask
            else:
                self.fallbacks += 1
        out = []
        for i, (ch, c) in enumerate(zip(bare, cls)):
            if unicodedata.category(ch).startswith("L"):
                mark = self.i2d[int(c)]
                if mark != "":
                    if self.conf and conf[i] < self.conf:
                        mark = ""
                    elif gate_mask is not None and not gate_mask[i]:
                        mark = ""
                out.append(ch + mark)
            else:
                out.append(ch)
        return unicodedata.normalize("NFC", "".join(out))


def evaluate(model, lines):
    agg = [0] * 6
    for gold in lines:
        for i, v in enumerate(score(model.diacritize(gold), gold)):
            agg[i] += v
    dw, dt, mw, mt, ww, wt = agg
    return dw / dt, mw / mt, ww / wt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--file", default=None)
    args = ap.parse_args()

    if args.file:
        path = args.file
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            "TigreGotico/arabic_diacritized_text", "test.txt", repo_type="dataset"
        )
    lines = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln and len(nfc_strip(ln)) <= 400:
                lines.append(ln)
            if len(lines) >= args.limit:
                break
    print(f"{len(lines)} test sentences\n")
    print(f"{'system':<28} {'DER(all)':>9} {'DER*(mk)':>9} {'WER':>8}")
    print("-" * 56)

    der, derm, wer = evaluate(Diacritizer("bilstm"), lines)
    print(f"{'bilstm (reference)':<28} {der:>8.2%} {derm:>8.2%} {wer:>7.2%}")
    der, derm, wer = evaluate(Diacritizer("rawi"), lines)
    print(f"{'rawi (no filter)':<28} {der:>8.2%} {derm:>8.2%} {wer:>7.2%}")
    print("-" * 56)

    t0 = time.time()
    gate = BilstmGate()
    configs = [
        ("rawi gated-by-bilstm", RawiGated(gate, 0.0)),
        ("rawi gated + conf>=0.5", RawiGated(gate, 0.5)),
        ("rawi gated + conf>=0.7", RawiGated(gate, 0.7)),
    ]
    for tag, model in configs:
        der, derm, wer = evaluate(model, lines)
        fb = f"  (fallbacks: {model.fallbacks})" if model.fallbacks else ""
        print(f"{tag:<28} {der:>8.2%} {derm:>8.2%} {wer:>7.2%}{fb}")
    print(f"\n({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
