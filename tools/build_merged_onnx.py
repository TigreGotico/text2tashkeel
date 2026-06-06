"""Stitch the `bilstm+rawi` gated ensemble into ONE ONNX graph.

The two models PLUS the agreement-gating logic become a single graph:

    inputs : b_input_ids (int64 [B,N])   — bilstm-tokenized text  (the gate)
             r_input     (int64 [B,N])   — rawi-tokenized text    (the value)
    output : gated_cls   (int64 [B,N])   — rawi's class, zeroed where bilstm marks nothing

i.e.  gated = where(argmax(bilstm) == 0, 0, argmax(rawi))

WHAT THIS DOES AND DOESN'T DO
-----------------------------
It folds the two LSTM forward passes and the gating math into one session — so you
do a single `session.run`, and the gate (argmax / equal / where) runs in the ONNX
runtime instead of Python.

It does NOT make a text-in / text-out model. ONNX has no Unicode normalization
(NFC/NFD) and no Unicode-category logic, so each model's *tokenization* (text -> the
two id sequences) and the final *detokenization* (class id -> diacritic string,
interleaved with the base letters) stay in host code. The two id sequences must be
the same length and position-aligned (bilstm uses NFC, rawi NFD, but both yield one
position per base letter — a length mismatch falls back, exactly as the Python
ensemble does).

Only `bilstm+rawi` (both opset 17) is merged here. The 3-model `ensemble` adds
libtashkeel, which is opset 16 (needs a version bump) and drops/normalizes some
input chars (variable length -> awkward to align inside the graph). Mergeable, but
fiddlier; the 2-model graph is the clean win.

    python tools/build_merged_onnx.py -o ensemble_bilstm_rawi.onnx
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, compose, helper, numpy_helper

from text2tashkeel import Diacritizer
from text2tashkeel._models import _BILSTM_C2I, _BILSTM_UNK, _MODELS_DIR, _strip


def build(out_path: str) -> None:
    b = compose.add_prefix(onnx.load(_MODELS_DIR / "bilstm.onnx"), prefix="b_")
    r = compose.add_prefix(onnx.load(_MODELS_DIR / "rawi.onnx"), prefix="r_")
    merged = compose.merge_models(b, r, io_map=[])      # parallel union
    g = merged.graph
    g.initializer.append(numpy_helper.from_array(np.array(0, np.int64), name="zero_i64"))
    g.node.extend([
        helper.make_node("ArgMax", ["b_logits"], ["b_cls"], axis=-1, keepdims=0),
        helper.make_node("ArgMax", ["r_output"], ["r_cls"], axis=-1, keepdims=0),
        helper.make_node("Equal", ["b_cls", "zero_i64"], ["is_bare"]),
        helper.make_node("Where", ["is_bare", "zero_i64", "r_cls"], ["gated_cls"]),
    ])
    del g.output[:]
    g.output.append(
        helper.make_tensor_value_info("gated_cls", TensorProto.INT64, ["batch", "seq"])
    )
    onnx.checker.check_model(merged)
    onnx.save(merged, out_path)
    print(f"wrote {out_path} ({Path(out_path).stat().st_size / 1e6:.1f} MB)")


def verify(out_path: str) -> None:
    rawi = Diacritizer("rawi").backend
    ref = Diacritizer("bilstm+rawi")
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])

    def merged_diac(text: str):
        rbare = "".join(
            c for c in rawi._normalize(text) if unicodedata.category(c) != "Mn"
        )
        bbare = _strip(text)
        if len(rbare) != len(bbare):
            return ref.diacritize(text)        # alignment fallback
        r_ids = np.array([[rawi.c2i.get(c, rawi.unk) for c in rbare]], np.int64)
        b_ids = np.array([[_BILSTM_C2I.get(c, _BILSTM_UNK) for c in bbare]], np.int64)
        gated = sess.run(["gated_cls"], {"r_input": r_ids, "b_input_ids": b_ids})[0][0]
        out = "".join(
            ch + (rawi.i2d[int(c)] if unicodedata.category(ch).startswith("L") else "")
            for ch, c in zip(rbare, gated)
        )
        return unicodedata.normalize("NFC", out)

    nfc = lambda s: unicodedata.normalize("NFC", s)
    ok = True
    for t in [
        "بسم الله الرحمن الرحيم", "العلم نور والجهل ظلام", "هذا كتاب مفيد",
        "في التأني السلامة وفي العجلة الندامة", "محمد رسول الله",
    ]:
        ok &= nfc(merged_diac(t)) == nfc(ref.diacritize(t))
    print("merged graph == Python bilstm+rawi:", ok)
    assert ok, "merged graph diverged from the Python ensemble"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="ensemble_bilstm_rawi.onnx")
    args = ap.parse_args()
    build(args.out)
    verify(args.out)


if __name__ == "__main__":
    main()
