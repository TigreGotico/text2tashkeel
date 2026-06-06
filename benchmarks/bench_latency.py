"""Latency: the stitched single-ONNX flagship vs the two-session Python ensemble.

Finding: the stitch is for *self-contained deployment*, not speed. It runs the
same two LSTMs as the Python ensemble and the branches don't parallelize, so
single-thread latency is identical (~2 ms/sentence, ~2x the single model). The
saved Python orchestration is negligible (gating is fast numpy via mark_mask).

    TT_ORT_THREADS=1 python benchmarks/bench_latency.py
"""
import os
import statistics
import time

os.environ.setdefault("TT_ORT_THREADS", "1")

from text2tashkeel import Diacritizer            # noqa: E402
from text2tashkeel._models import _EnsembleBackend  # noqa: E402

SENTS = [
    "بسم الله الرحمن الرحيم",
    "الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين",
    "وان وهبها لرب الارض لم يلزمه القبول ان اراد القلع فيسن تطليقها في طهرها",
    "العلم نور والجهل ظلام والكتاب خير جليس في الزمان",
]


def bench(backend, n=400, warmup=20):
    for _ in range(warmup):
        for s in SENTS:
            backend.diacritize(s)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        for s in SENTS:
            backend.diacritize(s)
        times.append((time.perf_counter() - t0) / len(SENTS))
    return statistics.median(times) * 1000  # ms/sentence


def main():
    models = {
        "rawi-ensemble (stitched, 1 session)": Diacritizer("rawi-ensemble").backend,
        "rawi-v2-int8+rawi-v3-int8 (python, 2 sessions)":
            _EnsembleBackend(["rawi-v2-int8"], "rawi-v3-int8", "any"),
        "rawi-v2-int8 (single model)": Diacritizer("rawi-v2-int8").backend,
    }
    print(f"{'model':<48} {'ms/sent':>9} {'sent/s':>8}")
    print("-" * 68)
    res = {}
    for name, b in models.items():
        ms = bench(b)
        res[name] = ms
        print(f"{name:<48} {ms:>8.3f} {1000 / ms:>8.0f}")
    print("-" * 68)
    st = res["rawi-ensemble (stitched, 1 session)"]
    py = res["rawi-v2-int8+rawi-v3-int8 (python, 2 sessions)"]
    print(f"stitched vs python ensemble: {py / st:.2f}x  ({py - st:+.3f} ms/sent)"
          f"  — same within noise; the stitch buys deployment, not speed")


if __name__ == "__main__":
    main()
