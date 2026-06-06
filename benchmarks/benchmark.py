"""Benchmark every bundled model on each split of a diacritized Arabic corpus.

Streams the raw split files of TigreGotico/arabic_diacritized_text (train / test
/ val) line by line, so the 3.4M-line corpus never has to fit in memory. Each
gold sentence is stripped to recover the model input, re-diacritized by every
model, and scored with DER / WER.

    python benchmarks/benchmark.py --limit 20000

  ⚠️  CONTAMINATION WARNING. This corpus is an aggregate of many public sources.
  Every model here was very likely trained on text that also appears in it (the
  rawi model definitely was — on a subset). So these numbers measure *fit on
  probably-seen data*, not generalization. Treat them as indicative, and do not
  read small gaps between models as a real quality ranking. See benchmarks/README.md.

Metrics (DER / WER) are vendored from the upstream diacritize.evaluate so the
numbers are comparable across models:
https://github.com/Z-Mahmood/arabic-diacritizer-public-release
"""

from __future__ import annotations

import argparse
import time
import unicodedata
from text2tashkeel import Diacritizer, available_models
from text2tashkeel._models import _STRIP

# ── Metrics (vendored from diacritize.evaluate, MIT, Z-Mahmood) ─────────────
_DIAC = {chr(c) for c in range(0x64B, 0x653)}
_SHADDA = "ّ"


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _strip(text: str) -> str:
    return "".join(c for c in _norm(text) if c not in _STRIP)


def extract_diacritics(text: str) -> list[str]:
    labels: list[str] = []
    current = ""
    for ch in _norm(text):
        if ch in _DIAC:
            current += ch
        else:
            if labels:
                labels[-1] = _reorder(current)
            labels.append("")
            current = ""
    if labels and current:
        labels[-1] = _reorder(current)
    return labels


def _reorder(d: str) -> str:
    if _SHADDA in d and len(d) > 1:
        return _SHADDA + d.replace(_SHADDA, "")
    return d


def score(pred: str, ref: str) -> tuple[int, int, int, int, int, int]:
    """Return six counts:
        der_wrong, der_total          — over ALL base characters (strict DER)
        mk_wrong,  mk_total           — over chars MARKED in gold only (lenient)
        wer_wrong, wer_total          — over whitespace-separated words
    `mk_*` mirrors rawi's training metric (mask = has-a-diacritic), so models
    that mostly err on already-easy no-mark positions are judged fairly too.
    """
    p, r = extract_diacritics(pred), extract_diacritics(ref)
    der_wrong = der_total = mk_wrong = mk_total = 0
    for a, b in zip(p, r):
        der_total += 1
        if a != b:
            der_wrong += 1
        if b != "":
            mk_total += 1
            if a != b:
                mk_wrong += 1
    der_total += len(r) - len(p) if len(r) > len(p) else 0  # unmatched gold = wrong
    der_wrong += len(r) - len(p) if len(r) > len(p) else 0
    pw, rw = _norm(pred).split(), _norm(ref).split()
    wer_wrong = sum(
        1 for a, b in zip(pw, rw) if extract_diacritics(a) != extract_diacritics(b)
    )
    return der_wrong, der_total, mk_wrong, mk_total, wer_wrong, len(rw)


# ── Corpus streaming ───────────────────────────────────────────────────────
def split_lines(split: str):
    """Yield gold sentences from a split, streaming from the cached file."""
    from huggingface_hub import hf_hub_download

    fname = {"train": "train.txt", "test": "test.txt", "val": "val.txt"}[split]
    path = hf_hub_download(
        "TigreGotico/arabic_diacritized_text", fname, repo_type="dataset"
    )
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line


# ── Benchmark ──────────────────────────────────────────────────────────────
def run_split(split: str, models: dict, limit: int, max_len: int) -> dict:
    keys = ("dw", "dt", "mw", "mt", "ww", "wt")
    stats = {name: dict.fromkeys(keys, 0) for name in models}
    n = 0
    t0 = time.time()
    for gold in split_lines(split):
        if len(_strip(gold)) > max_len:
            continue
        for name, model in models.items():   # sequential: less ORT/GIL contention
            s = stats[name]
            for k, v in zip(keys, score(model.diacritize(gold), gold)):
                s[k] += v
        n += 1
        if n % 2000 == 0:
            print(f"  [{split}] {n} ({n / (time.time() - t0):.0f}/s)", flush=True)
        if limit > 0 and n >= limit:
            break
    return {"n": n, "secs": time.time() - t0, "stats": stats}


def print_table(split: str, result: dict) -> None:
    print(f"\n=== split: {split}  ({result['n']} sentences, {result['secs']:.0f}s) ===")
    print(f"{'model':<14} {'DER':>8} {'DER*':>8} {'WER':>8}   {'chars':>9} {'words':>9}")
    print(f"{'':14} {'(all)':>8} {'(marked)':>8}")
    print("-" * 62)
    for name, s in result["stats"].items():
        der = s["dw"] / s["dt"] if s["dt"] else 0.0
        derm = s["mw"] / s["mt"] if s["mt"] else 0.0
        wer = s["ww"] / s["wt"] if s["wt"] else 0.0
        print(f"{name:<14} {der:>7.2%} {derm:>7.2%} {wer:>7.2%}   {s['dt']:>9} {s['wt']:>9}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20000,
                    help="max sentences per split (-1 = full split)")
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--splits", nargs="+", default=["test", "val", "train"])
    ap.add_argument("--models", nargs="+", default=available_models())
    args = ap.parse_args()

    print("loading models:", args.models, flush=True)
    models = {m: Diacritizer(m) for m in args.models}
    for d in models.values():       # warm the sessions
        d.diacritize("اختبار")

    results = {}
    for split in args.splits:
        results[split] = run_split(split, models, args.limit, args.max_len)
        print_table(split, results[split])


if __name__ == "__main__":
    main()
