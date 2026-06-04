"""06 — Drive the ONNX session directly (no wrapper).

Useful if you're porting to another language or want the raw logits. This
mirrors what the bilstm backend does internally.

Run:  python examples/06_raw_onnx.py
"""

import numpy as np

from text2tashkeel import Diacritizer
from text2tashkeel._models import (
    _BILSTM_C2I as C2I,
    _BILSTM_ID2LABEL as ID2LABEL,
    _BILSTM_UNK as UNK,
    _strip,
)

# Get the raw onnxruntime.InferenceSession from the backend.
sess = Diacritizer("bilstm").backend.sess
print("inputs :", [(i.name, i.type, i.shape) for i in sess.get_inputs()])
print("outputs:", [(o.name, o.type, o.shape) for o in sess.get_outputs()])
print()

text = "هذا كتاب"
bare = _strip(text)

# Encode: characters -> int64 IDs, shape (batch=1, seq_len).
input_ids = np.array([[C2I.get(c, UNK) for c in bare]], dtype=np.int64)

# Run: logits shape (1, seq_len, 15).
logits = sess.run(["logits"], {"input_ids": input_ids})[0]
print("logits shape:", logits.shape)

# Decode: argmax over the 15 classes, map each to its mark, re-attach.
pred = logits[0].argmax(-1)
out = "".join(ch + ID2LABEL[int(c)] for ch, c in zip(bare, pred))
print("result      :", out)
