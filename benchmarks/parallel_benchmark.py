"""Full-file benchmark, parallelized across cores (process pool).

For scoring an entire split (e.g. the 820k-line test.txt) on a many-core machine.
Self-contained: reads a local text file directly (no datasets/HF), one diacritized
sentence per line, and imports the local text2tashkeel package.

CRUCIAL for speed: pin BOTH onnxruntime AND BLAS to one thread per worker, then
parallelize with the process pool. Without this, numpy/OpenBLAS spawns a full
thread pool *per worker* and the machine thrashes (measured 80/s thrashing vs
207/s pinned, on a busy 24-core box). Always launch like:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        nice -n 15 python benchmarks/parallel_benchmark.py --file test.txt --workers 12

(The script also sets TT_ORT_THREADS=1 itself.) Keep --workers well below the core
count on a shared box — leave headroom for whatever else it serves.
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

from text2tashkeel import available_models  # noqa: E402

# ── metrics (vendored; identical to benchmarks/benchmark.py) ────────────────
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


# ── worker ─────────────────────────────────────────────────────────────────
_MODELS = None
_KEYS = ("dw", "dt", "mw", "mt", "ww", "wt")


def _init(model_names):
    global _MODELS
    from text2tashkeel import Diacritizer

    _MODELS = {m: Diacritizer(m) for m in model_names}
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


MAX_LEN = 400  # set in main, re-broadcast via globals to workers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=-1, help="-1 = full file")
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--file", default="test.txt")
    ap.add_argument("--out", default="results_full_test.txt")
    ap.add_argument("--models", nargs="+", default=available_models())
    args = ap.parse_args()

    global MAX_LEN
    MAX_LEN = args.max_len

    print(f"reading {args.file} ...", flush=True)
    with open(args.file, encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    if args.limit > 0:
        lines = lines[: args.limit]
    total = len(lines)
    chunks = [lines[i : i + args.chunk] for i in range(0, total, args.chunk)]
    print(f"{total} sentences in {len(chunks)} chunks · {args.workers} workers "
          f"· models={args.models}", flush=True)

    agg = {name: dict.fromkeys(_KEYS, 0) for name in args.models}
    done = 0
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(args.models,)) as pool:
        for n, stats in pool.imap_unordered(_work, chunks):
            done += n
            for name in args.models:
                for k in _KEYS:
                    agg[name][k] += stats[name][k]
            rate = done / (time.time() - t0)
            print(f"  {done}/{total}  ({rate:.0f}/s, "
                  f"eta {(total-done)/rate/60:.1f} min)", flush=True)

    dur = time.time() - t0
    lines_out = []
    lines_out.append(f"FULL test.txt benchmark — {done} sentences in {dur/60:.1f} min "
                     f"({done/dur:.0f}/s, {args.workers} workers)")
    lines_out.append("")
    lines_out.append(f"{'model':<14} {'DER(all)':>9} {'DER*(mk)':>9} {'WER':>8} "
                     f"{'chars':>11} {'words':>11}")
    lines_out.append("-" * 66)
    for name in args.models:
        s = agg[name]
        der = s["dw"] / s["dt"] if s["dt"] else 0
        derm = s["mw"] / s["mt"] if s["mt"] else 0
        wer = s["ww"] / s["wt"] if s["wt"] else 0
        lines_out.append(f"{name:<14} {der:>8.2%} {derm:>8.2%} {wer:>7.2%} "
                         f"{s['dt']:>11} {s['wt']:>11}")
    report = "\n".join(lines_out)
    print("\n" + report)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
