"""Stitch the `rawi-v2 + rawi-v3` gated ensemble into ONE single-input ONNX.

rawi-v2 (the gate, decides WHERE) and rawi-v3 (the value head, decides WHICH)
share the *same* vocab and NFD normalization, so a single token-id sequence feeds
both — unlike bilstm+rawi (two vocabs → two inputs). The graph is:

    input : input (int64 [B,N])   — rawi-tokenized ids (shared)
    output: gated_cls (int64 [B,N]) — v3 value class, zeroed where v2 marks nothing

    gated = where(argmax(v2) == 0, 0, argmax(v3.value))

Tokenization (text → ids) and detokenization (class → mark string, applied to
letters) stay in host code — ONNX has no Unicode normalization or category logic.
One `session.run`, one file, one input.

    python tools/build_ensemble_v2v3_onnx.py -o ensemble_rawi_v2v3.onnx
"""
from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, compose, helper, numpy_helper

from text2tashkeel._models import _MODELS_DIR, _EnsembleBackend


def build(out_path: str, int8: bool = False) -> None:
    gate_f = "rawi_v2.int8.onnx" if int8 else "rawi_v2.onnx"
    val_f = "rawi_v3.int8.onnx" if int8 else "rawi_v3.onnx"
    g = compose.add_prefix(onnx.load(_MODELS_DIR / gate_f), prefix="g_")
    v = compose.add_prefix(onnx.load(_MODELS_DIR / val_f), prefix="v_")
    merged = compose.merge_models(g, v, io_map=[])     # parallel union (g_input, v_input)
    graph = merged.graph

    # collapse the two inputs into one shared "input" via Identity fan-out
    del graph.input[:]
    graph.input.append(
        helper.make_tensor_value_info("input", TensorProto.INT64, ["batch", "seq"])
    )
    graph.initializer.append(numpy_helper.from_array(np.array(0, np.int64), "zero_i64"))
    # Identity fan-out must precede the subgraphs that consume g_input/v_input.
    graph.node.insert(0, helper.make_node("Identity", ["input"], ["v_input"]))
    graph.node.insert(0, helper.make_node("Identity", ["input"], ["g_input"]))
    # gating math runs after both subgraphs produce their logits
    graph.node.extend([
        # gate = rawi-v2 argmax; value = rawi-v3 *value* head argmax
        helper.make_node("ArgMax", ["g_output"], ["g_cls"], axis=-1, keepdims=0),
        helper.make_node("ArgMax", ["v_value"], ["v_cls"], axis=-1, keepdims=0),
        helper.make_node("Equal", ["g_cls", "zero_i64"], ["is_bare"]),
        helper.make_node("Where", ["is_bare", "zero_i64", "v_cls"], ["gated_cls"]),
    ])
    del graph.output[:]
    graph.output.append(
        helper.make_tensor_value_info("gated_cls", TensorProto.INT64, ["batch", "seq"])
    )
    onnx.checker.check_model(merged)
    onnx.save(merged, out_path)
    print(f"wrote {out_path} ({Path(out_path).stat().st_size / 1e6:.1f} MB)")


def verify(out_path: str, int8: bool = False) -> None:
    gate, value = ("rawi-v2-int8", "rawi-v3-int8") if int8 else ("rawi-v2", "rawi-v3")
    ref = _EnsembleBackend([gate], value, "any")   # the Python ensemble
    val = ref.value                                 # rawi-v3 backend (vocab + i2d)
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    nfc = lambda s: unicodedata.normalize("NFC", s)

    def merged_diac(text: str) -> str:
        nfd = "".join(
            c for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "So"
        )
        bare = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
        if not bare:
            return text
        ids = np.array([[val.c2i.get(c, val.unk) for c in bare]], np.int64)
        cls = sess.run(["gated_cls"], {"input": ids})[0][0]
        out = "".join(
            ch + (val.i2d[int(c)] if unicodedata.category(ch).startswith("L") else "")
            for ch, c in zip(bare, cls)
        )
        return nfc(out)

    tests = [
        "بسم الله الرحمن الرحيم", "العلم نور والجهل ظلام", "هذا كتاب مفيد",
        "في التأني السلامة وفي العجلة الندامة", "محمد رسول الله",
        "الحمد لله رب العالمين", "وإن وهبها لرب الأرض لم يلزمه القبول",
    ]
    ok = all(merged_diac(t) == nfc(ref.diacritize(t)) for t in tests)
    print(f"merged graph == Python {gate}+{value}:", ok)
    assert ok, "merged graph diverged from the Python ensemble"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="ensemble_rawi_v2v3.onnx")
    ap.add_argument("--int8", action="store_true", help="stitch the int8 models (~5 MB)")
    args = ap.parse_args()
    build(args.out, args.int8)
    verify(args.out, args.int8)


if __name__ == "__main__":
    main()
