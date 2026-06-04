"""05 — Diacritize many sentences correctly.

The safe, correct way to process a list is ONE sentence per call. This script
also demonstrates *why* you must not pad-and-stack variable-length sentences
into a single batch: the model's attention is unmasked, so padding leaks into
the predictions and changes them.

Run:  python examples/05_batch_safely.py
"""

import numpy as np

from text2tashkeel import Diacritizer
from text2tashkeel._models import _BILSTM_C2I as C2I, _BILSTM_UNK as UNK, _strip

d = Diacritizer("bilstm")

# ── The correct pattern: loop, one sentence per call. ──────────────────────
corpus = [
    "هذا كتاب مفيد",
    "العلم نور",
    "بسم الله الرحمن الرحيم",
]
print("Correct (per-sentence):")
for s in corpus:
    print("  ", d.diacritize(s))

# For throughput, run independent calls in threads — onnxruntime releases the
# GIL during inference. (Shown small here; scale with concurrent.futures.)

# ── Why NOT to pad-and-batch: a demonstration. ─────────────────────────────
print("\nWhy padding is unsafe (same inputs, with vs without padding to length 80):")
sess = d.backend.sess
total = flipped = 0
for s in ["هذا", "العلم نور", "من", "كتاب", "قال له"]:
    ids = [C2I.get(c, UNK) for c in _strip(s)]
    a = sess.run(["logits"], {"input_ids": np.array([ids], np.int64)})[0][0].argmax(-1)
    padded = ids + [0] * (80 - len(ids))            # what naive batching would do
    b = sess.run(["logits"], {"input_ids": np.array([padded], np.int64)})[0][0][: len(ids)].argmax(-1)
    flips = int((a != b).sum())
    total += len(ids); flipped += flips
    print(f"  {s:<12} {flips}/{len(ids)} letters changed")
print(f"  TOTAL: {flipped}/{total} predicted marks changed just from padding.")
print("  => never pad-and-stack different-length sentences; process them separately.")
