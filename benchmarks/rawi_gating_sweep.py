"""Fine sweep of gate choice x confidence threshold for the gated rawi ensemble.

Efficient: each model is run ONCE per sentence (rawi logits, bilstm output,
libtashkeel output) and cached; then every (gate, threshold) combination is
evaluated by cheap pure-Python reconstruction — so we can scan a fine grid over a
large sample without re-running ONNX.

Gates tested: bilstm, libtashkeel, AND (both mark), OR (either marks).
For the bilstm gate we also sweep confidence>=t on the kept marks.

    python benchmarks/rawi_gating_sweep.py --limit 8000
"""

from __future__ import annotations

import argparse
import time
import unicodedata

import numpy as np

from text2tashkeel import Diacritizer
from text2tashkeel._models import _strip as nfc_strip

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8000)
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
    print(f"{len(lines)} test sentences")

    bilstm = Diacritizer("bilstm")
    libt = Diacritizer("libtashkeel")
    rawi = Diacritizer("rawi").backend
    i2d = rawi.i2d

    # ── precompute per-sentence model outputs (one ONNX run each) ───────────
    print("precomputing model outputs ...", flush=True)
    t0 = time.time()
    cache = []
    for gold in lines:
        bare, cls = rawi._predict(gold)
        if not bare:
            cache.append(None)
            continue
        ids = np.array([[rawi.c2i.get(c, rawi.unk) for c in bare]], dtype=np.int64)
        logits = rawi.sess.run(["output"], {"input": ids})[0][0]
        m = logits.max(-1, keepdims=True)
        e = np.exp(logits - m)
        conf = (e / e.sum(-1, keepdims=True))[np.arange(len(cls)), cls]
        bmask = np.array([l != "" for l in extract(bilstm.diacritize(gold))])
        lmask = np.array([l != "" for l in extract(libt.diacritize(gold))])
        is_letter = np.array([unicodedata.category(c).startswith("L") for c in bare])
        cache.append((gold, bare, cls, conf, is_letter, bmask, lmask))
    print(f"  ({time.time() - t0:.0f}s)\n")

    def reconstruct(item, gate, conf_th):
        gold, bare, cls, conf, is_letter, bmask, lmask = item
        n = len(bare)
        if gate == "bilstm":
            gm = bmask if len(bmask) == n else None
        elif gate == "libtashkeel":
            gm = lmask if len(lmask) == n else None
        elif gate == "and":
            gm = (bmask & lmask) if (len(bmask) == n and len(lmask) == n) else None
        elif gate == "or":
            gm = (bmask | lmask) if (len(bmask) == n and len(lmask) == n) else None
        else:
            gm = None
        out = []
        for i in range(n):
            ch = bare[i]
            if is_letter[i]:
                mark = i2d[int(cls[i])]
                if mark:
                    if conf_th and conf[i] < conf_th:
                        mark = ""
                    elif gm is not None and not gm[i]:
                        mark = ""
                out.append(ch + mark)
            else:
                out.append(ch)
        return unicodedata.normalize("NFC", "".join(out))

    def eval_config(gate, conf_th):
        agg = [0] * 6
        for item in cache:
            if item is None:
                continue
            for i, v in enumerate(score(reconstruct(item, gate, conf_th), item[0])):
                agg[i] += v
        dw, dt, mw, mt, ww, wt = agg
        return dw / dt, mw / mt, ww / wt

    # references
    print(f"{'config':<34} {'DER(all)':>9} {'DER*(mk)':>9} {'WER':>8}")
    print("-" * 62)
    for tag, mdl in [("bilstm", bilstm), ("rawi (raw)", Diacritizer("rawi"))]:
        agg = [0] * 6
        for item in cache:
            if item is None:
                continue
            for i, v in enumerate(score(mdl.diacritize(item[0]), item[0])):
                agg[i] += v
        print(f"{tag:<34} {agg[0]/agg[1]:>8.2%} {agg[2]/agg[3]:>8.2%} {agg[4]/agg[5]:>7.2%}")
    print("-" * 62)

    # gate comparison (no confidence)
    for gate in ["bilstm", "libtashkeel", "and", "or"]:
        der, derm, wer = eval_config(gate, 0.0)
        print(f"{'gate=' + gate:<34} {der:>8.2%} {derm:>8.2%} {wer:>7.2%}")
    print("-" * 62)

    # fine confidence sweep on the bilstm gate
    for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        der, derm, wer = eval_config("bilstm", th)
        print(f"{'gate=bilstm + conf>=' + format(th, 'g'):<34} {der:>8.2%} {derm:>8.2%} {wer:>7.2%}")


if __name__ == "__main__":
    main()
