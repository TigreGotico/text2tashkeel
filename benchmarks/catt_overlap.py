"""Quantify training-data overlap between CATT's corpus and ours.

The contamination question for the SOTA comparison: did CATT train on sentences
that appear in *our* test split? If so, CATT's score on our benchmark is inflated
in a way rawi-v2's is not (rawi-v2 provably never saw our test.txt). We also check
the converse and the shared-source overlap, and emit plots.

Matching is on the **bare** (undiacritized, whitespace-collapsed, NFC) sentence —
the contamination-relevant key: same sentence seen, regardless of how it was
marked. Sets are deduped.
"""
import json
import sys
import unicodedata
from pathlib import Path

OUR = Path.home() / (
    ".cache/huggingface/hub/datasets--TigreGotico--arabic_diacritized_text/"
    "snapshots/b9ac17960a5d01bd110267a0b736ab844ad30f71"
)
CATT = Path("/home/miro/AgentWorkspaces/.scratch/catt/dataset")
OUTDIR = Path(__file__).resolve().parent

_STRIP = {chr(c) for c in range(0x64B, 0x653)} | {"ٰ"}


def key(line: str) -> int:
    s = unicodedata.normalize("NFC", line.strip())
    bare = "".join(c for c in s if c not in _STRIP)
    bare = " ".join(bare.split())
    return hash(bare)


def load_keys(path: Path):
    ks = set()
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                ks.add(key(ln))
    return ks


def stream_overlap(path: Path, *refsets):
    """Stream a (possibly huge) file; count line membership in each ref set.
    Returns (n_lines, n_unique, [hits_per_refset], unique_keys_set_or_None)."""
    counts = [0] * len(refsets)
    n = 0
    seen = set()
    uhits = [0] * len(refsets)
    for ln in open(path, encoding="utf-8"):
        if not ln.strip():
            continue
        n += 1
        k = key(ln)
        for i, rs in enumerate(refsets):
            if k in rs:
                counts[i] += 1
        if k not in seen:
            seen.add(k)
            for i, rs in enumerate(refsets):
                if k in rs:
                    uhits[i] += 1
    return n, len(seen), counts, uhits


def main():
    print("loading our test + train key sets ...", flush=True)
    our_test = load_keys(OUR / "test.txt")
    our_train = load_keys(OUR / "train.txt")
    print(f"  our test unique:  {len(our_test):>9}")
    print(f"  our train unique: {len(our_train):>9}")

    print("streaming CATT train.txt (1 GB) vs our test & train ...", flush=True)
    ct_n, ct_u, ct_hits, ct_uhits = stream_overlap(
        CATT / "train" / "train.txt", our_test, our_train
    )
    print(f"  CATT train lines: {ct_n}  unique: {ct_u}")

    print("loading CATT test/val ...", flush=True)
    catt_test = load_keys(CATT / "test" / "test.txt")
    catt_val = load_keys(CATT / "val" / "val.txt")

    R = {
        "our_test_unique": len(our_test),
        "our_train_unique": len(our_train),
        "catt_train_lines": ct_n,
        "catt_train_unique": ct_u,
        "catt_test_unique": len(catt_test),
        "catt_val_unique": len(catt_val),
        # THE key number: our test sentences present in CATT's training data
        "our_test_in_catt_train_unique": ct_uhits[0],
        "our_test_in_catt_train_pct": 100 * ct_uhits[0] / len(our_test),
        # sanity: our test sentences in OUR train (rawi-v2's exposure) — want ~0
        "our_test_in_our_train_unique": len(our_test & our_train),
        "our_test_in_our_train_pct": 100 * len(our_test & our_train) / len(our_test),
        # shared-source overlap between the two training corpora
        "our_train_in_catt_train_unique": ct_uhits[1],
        "our_train_in_catt_train_pct": 100 * ct_uhits[1] / len(our_train),
        # do the test sets coincide / leak into the other's train?
        "catt_test_in_our_train": len(catt_test & our_train),
        "catt_test_in_our_test": len(catt_test & our_test),
        "catt_test_in_our_train_pct": 100 * len(catt_test & our_train) / len(catt_test),
        "catt_val_in_our_train": len(catt_val & our_train),
    }
    (OUTDIR / "catt_overlap.json").write_text(json.dumps(R, indent=2))
    print("\n=== OVERLAP METRICS ===")
    for k, v in R.items():
        print(f"  {k:<34} {v:>12.2f}" if isinstance(v, float) else f"  {k:<34} {v:>12}")

    # ── plots ───────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        imgs = OUTDIR.parent / "docs" / "images"
        imgs.mkdir(parents=True, exist_ok=True)

        # 1) contamination bars: % of each "test" that leaked into a training set
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [
            "our test\nin CATT train",
            "our test\nin our train\n(rawi-v2 exposure)",
            "CATT test\nin our train",
        ]
        vals = [R["our_test_in_catt_train_pct"],
                R["our_test_in_our_train_pct"],
                R["catt_test_in_our_train_pct"]]
        colors = ["#d62728", "#2ca02c", "#1f77b4"]
        bars = ax.bar(labels, vals, color=colors)
        ax.set_ylabel("% of sentences present in the training set")
        ax.set_title("Training-data contamination (bare-sentence match)")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(imgs / "catt_contamination.png", dpi=120)

        # 2) corpus size comparison (unique sentences)
        fig, ax = plt.subplots(figsize=(7, 4))
        names = ["our train", "CATT train", "our test", "CATT test"]
        sizes = [R["our_train_unique"], R["catt_train_unique"],
                 R["our_test_unique"], R["catt_test_unique"]]
        ax.bar(names, sizes, color=["#2ca02c", "#ff7f0e", "#2ca02c", "#ff7f0e"])
        ax.set_yscale("log")
        ax.set_ylabel("unique sentences (log)")
        ax.set_title("Corpus sizes")
        for i, s in enumerate(sizes):
            ax.text(i, s, f"{s:,}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(imgs / "catt_corpus_sizes.png", dpi=120)
        print(f"\nplots → {imgs}/catt_contamination.png, catt_corpus_sizes.png")
    except Exception as e:  # noqa
        print(f"plotting skipped: {e}")


if __name__ == "__main__":
    sys.exit(main())
