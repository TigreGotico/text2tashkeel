"""Experiment: does gating rawi-v2 (best at WHERE) with rawi-v1 as the value
model (best at WHICH mark) beat rawi-v2 standalone?

rawi-v1 has the lowest DER* (marked-position error, 3.07%) of any model — it
picks the right mark when there *is* one — but it over-marks, so it's useless
alone. rawi-v2 has the best WHERE calibration (DER 2.29%) but a slightly higher
DER* (3.37%). The hypothesis: use v2 to decide where, v1 to decide which.

Self-contained, parallel, full-file. Compares standalone vs the gated combo so
we see the delta directly. Run like parallel_benchmark.py.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import unicodedata
from multiprocessing import Pool

os.environ["TT_ORT_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# ── metrics (identical to parallel_benchmark.py) ────────────────────────────
_STRIP = {chr(c) for c in range(0x64B, 0x653)} | {"ٰ"}
_DIAC = {chr(c) for c in range(0x64B, 0x653)}
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


# ── models built directly (no registry) ────────────────────────────────────
_MODELS = None
_KEYS = ("dw", "dt", "mw", "mt", "ww", "wt")
MAX_LEN = 400


def _init(_):
    global _MODELS
    from text2tashkeel._models import build_backend, _EnsembleBackend
    _MODELS = {
        "rawi-v2-int8": build_backend("rawi-v2-int8"),             # default ref
        "rawi-v2+rawi-int8": _EnsembleBackend(["rawi-v2"], "rawi-int8", "any"),
        "rawi-v2-int8+rawi-int8": _EnsembleBackend(["rawi-v2-int8"], "rawi-int8", "any"),
    }
    for d in _MODELS.values():
        d.diacritize("اختبار")


def _work(chunk):
    stats = {name: dict.fromkeys(_KEYS, 0) for name in _MODELS}
    n = 0
    for gold in chunk:
        if len(_strip(gold)) > MAX_LEN:
            continue
        for name, d in _MODELS.items():
            s = stats[name]
            for k, v in zip(_KEYS, score(d.diacritize(gold), gold)):
                s[k] += v
        n += 1
    return n, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--file", default="test.txt")
    ap.add_argument("--out", default="results_v2_gate.txt")
    args = ap.parse_args()

    with open(args.file, encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    if args.limit > 0:
        lines = lines[: args.limit]
    total = len(lines)
    chunks = [lines[i:i + args.chunk] for i in range(0, total, args.chunk)]
    print(f"{total} sentences · {args.workers} workers", flush=True)

    names = ["rawi-v2-int8", "rawi-v2+rawi-int8", "rawi-v2-int8+rawi-int8"]
    agg = {name: dict.fromkeys(_KEYS, 0) for name in names}
    done, t0 = 0, time.time()
    with Pool(args.workers, initializer=_init, initargs=(None,)) as pool:
        for n, stats in pool.imap_unordered(_work, chunks):
            done += n
            for name in names:
                for k in _KEYS:
                    agg[name][k] += stats[name][k]
            rate = done / (time.time() - t0)
            print(f"  {done}/{total} ({rate:.0f}/s, eta {(total-done)/rate/60:.1f} min)",
                  flush=True)

    dur = time.time() - t0
    out = [f"v2-gate experiment — {done} sentences in {dur/60:.1f} min",
           "",
           f"{'model':<20} {'DER(all)':>9} {'DER*(mk)':>9} {'WER':>8}",
           "-" * 50]
    for name in names:
        s = agg[name]
        out.append(f"{name:<20} {s['dw']/s['dt']:>8.2%} {s['mw']/s['mt']:>8.2%} "
                   f"{s['ww']/s['wt']:>7.2%}")
    report = "\n".join(out)
    print("\n" + report)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")


if __name__ == "__main__":
    main()
