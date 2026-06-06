"""Experiment: does confidence-thresholding rawi's predictions cut its over-marking?

Hypothesis: rawi has a low DER*(marked) but a high DER(all)/WER because it sprays
spurious, *low-confidence* marks onto positions the gold leaves bare. If so,
suppressing any predicted mark whose softmax probability is below a threshold
(emit "no mark" instead) should lower DER(all) and WER — ideally without wrecking
DER*(marked).

Sweeps the threshold and prints DER(all) / DER*(marked) / WER for each, with
bilstm as a reference line.

    python benchmarks/rawi_confidence_filter.py --limit 5000
"""

from __future__ import annotations

import argparse
import time
import unicodedata

import numpy as np

from text2tashkeel import Diacritizer

# ── metric (same as benchmarks/benchmark.py) ───────────────────────────────
_DIAC = {chr(c) for c in range(0x64B, 0x653)}
_STRIP = _DIAC | {"ٰ"}
_SHADDA = "ّ"


def _norm(s):
    return unicodedata.normalize("NFC", s)


def _strip(s):
    return "".join(c for c in _norm(s) if c not in _STRIP)


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


# ── rawi with a confidence filter ──────────────────────────────────────────
class RawiFiltered:
    """mode='conf'  : suppress a predicted mark if its softmax prob < threshold.
    mode='margin' : suppress a predicted mark if the 'no-mark' (empty) class is a
                    close runner-up, i.e. p(empty) >= p(mark) - threshold.
    The empty class id is the one whose i2d string is ''."""

    def __init__(self, threshold: float, mode: str = "conf"):
        b = Diacritizer("rawi").backend
        self.sess, self.i2d, self.c2i, self.unk = b.sess, b.i2d, b.c2i, b.unk
        self.norm = b._normalize
        self.th = threshold
        self.mode = mode
        self.empty_id = next(i for i, s in self.i2d.items() if s == "")

    def diacritize(self, text: str) -> str:
        norm = self.norm(text)
        bare = "".join(c for c in norm if unicodedata.category(c) != "Mn")
        if not bare:
            return text
        ids = np.array([[self.c2i.get(c, self.unk) for c in bare]], dtype=np.int64)
        logits = self.sess.run(["output"], {"input": ids})[0][0]  # (seq, 73)
        m = logits.max(-1, keepdims=True)
        e = np.exp(logits - m)
        probs = e / e.sum(-1, keepdims=True)
        cls = logits.argmax(-1)
        conf = probs[np.arange(len(cls)), cls]
        p_empty = probs[:, self.empty_id]
        out = []
        for ch, c, cf, pe in zip(bare, cls, conf, p_empty):
            if unicodedata.category(ch).startswith("L"):
                mark = self.i2d[int(c)]
                if mark != "":
                    drop = cf < self.th if self.mode == "conf" else pe >= cf - self.th
                    if drop:
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
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--file", default=None, help="test.txt (defaults to HF cache)")
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99])
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
            if ln and len(_strip(ln)) <= 400:
                lines.append(ln)
            if len(lines) >= args.limit:
                break
    print(f"{len(lines)} test sentences\n")

    print(f"{'system':<22} {'DER(all)':>9} {'DER*(mk)':>9} {'WER':>8}")
    print("-" * 50)

    der, derm, wer = evaluate(Diacritizer("bilstm"), lines)
    print(f"{'bilstm (reference)':<22} {der:>8.2%} {derm:>8.2%} {wer:>7.2%}")
    print("-" * 50)

    t0 = time.time()
    for th in args.thresholds:
        der, derm, wer = evaluate(RawiFiltered(th, "conf"), lines)
        tag = "rawi (no filter)" if th == 0.0 else f"rawi conf>={th:g}"
        print(f"{tag:<22} {der:>8.2%} {derm:>8.2%} {wer:>7.2%}")
    print("-" * 50)
    for mg in [0.0, 0.1, 0.2, 0.3]:
        der, derm, wer = evaluate(RawiFiltered(mg, "margin"), lines)
        print(f"{'rawi margin<=' + format(mg, 'g'):<22} {der:>8.2%} {derm:>8.2%} {wer:>7.2%}")
    print(f"\n({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
