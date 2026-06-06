"""03 — Watch the model think, one letter at a time.

Prints the four stages of the pipeline for the `bilstm` model (a simple
single-model path that maps each letter to one of 15 diacritic classes):
    1. the bare input (diacritics stripped)
    2. the integer ID for each letter
    3. the predicted diacritic CLASS for each letter (name + mark)
    4. the reassembled, diacritized string

This makes the "one decision per base letter" structure concrete.

Run:  python examples/03_understand_output.py
"""

import numpy as np

from text2tashkeel import Diacritizer
from text2tashkeel._models import (
    _BILSTM_C2I as C2I,
    _BILSTM_ID2LABEL as ID2LABEL,
    _BILSTM_UNK as UNK,
    _strip,
)

CLASS_NAMES = {
    0: "(none)", 1: "fatha /a/", 2: "damma /u/", 3: "kasra /i/", 4: "sukun (no vowel)",
    5: "shadda", 6: "fathatan /an/", 7: "dammatan /un/", 8: "kasratan /in/",
    9: "shadda+fatha", 10: "shadda+damma", 11: "shadda+kasra",
    12: "shadda+fathatan", 13: "shadda+dammatan", 14: "shadda+kasratan",
}

text = "هذا كتاب"
d = Diacritizer("bilstm")

# Stage 1: strip to the bare skeleton.
bare = _strip(text)
print("1. input (stripped):", bare, f"({len(bare)} characters)\n")

# Stage 2: letters -> integer IDs.
ids = [C2I.get(c, UNK) for c in bare]
print("2. token IDs:")
for ch, i in zip(bare, ids):
    shown = repr(ch) if ch != " " else "' '"
    print(f"     {shown:>5} -> {i}")

# Stage 3: run the model, read the predicted class per letter.
logits = d.backend.sess.run(["logits"], {"input_ids": np.array([ids], np.int64)})[0][0]
pred = logits.argmax(-1)
print("\n3. predicted diacritic class per letter:")
for ch, cls in zip(bare, pred):
    shown = repr(ch) if ch != " " else "' '"
    mark = ID2LABEL[int(cls)] or "—"
    print(f"     {shown:>5} -> class {int(cls):>2}  {CLASS_NAMES[int(cls)]:<16} mark: {mark}")

# Stage 4: reassemble.
print("\n4. output:", d.diacritize(text))
