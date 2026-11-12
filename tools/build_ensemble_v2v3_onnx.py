"""Stitch the `rawi-v2 + rawi-v3` gated ensemble into ONE single-input ONNX.

rawi-v2 (the gate, decides WHERE) and rawi-v3 (the value head, decides WHICH)
share the *same* vocab and NFD normalization, so a single token-id sequence feeds
both — unlike bilstm+rawi (two vocabs → two inputs). The graph is:

    input : input (int64 [B,N])   — rawi-tokenized ids (shared)
    output: gated_cls (int64 [B,N]) — v3 value class, zeroed where v2 marks nothing
            gated_logits (float [B,N,C]) — the distribution gated_cls is the argmax of

    gated        = where(argmax(v2) == 0, 0, argmax(v3.value))
    gated_logits = v3.value, with class 0 lifted to (row max + GATE_MARGIN)
                   where argmax(v2) == 0

so argmax(gated_logits) == gated_cls at every position, and every other class
keeps the value head's own score.

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

#: How far class 0 is lifted above the row max where the gate is bare: enough to
#: win the argmax, and no change to the scores of the other classes.
GATE_MARGIN = 1.0


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
    value_out = next(o for o in graph.output if o.name == "v_value")
    n_cls = value_out.type.tensor_type.shape.dim[-1].dim_value
    onehot0 = np.zeros(n_cls, np.float32)
    onehot0[0] = 1.0
    graph.initializer.extend([
        numpy_helper.from_array(np.array(0, np.int64), "zero_i64"),
        numpy_helper.from_array(np.array([0], np.int64), "idx0_i64"),
        numpy_helper.from_array(np.array([2], np.int64), "axis2_i64"),
        numpy_helper.from_array(np.array(GATE_MARGIN, np.float32), "gate_margin"),
        numpy_helper.from_array(onehot0, "onehot0"),
    ])
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
        # gated_logits = v_value + boost on class 0, where the gate is bare
        helper.make_node("ReduceMax", ["v_value"], ["v_max"], axes=[-1], keepdims=1),
        helper.make_node("Gather", ["v_value", "idx0_i64"], ["v_at0"], axis=-1),
        helper.make_node("Sub", ["v_max", "v_at0"], ["v_deficit"]),
        helper.make_node("Add", ["v_deficit", "gate_margin"], ["v_deficit_m"]),
        helper.make_node("Relu", ["v_deficit_m"], ["v_boost"]),
        helper.make_node("Cast", ["is_bare"], ["is_bare_f"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["is_bare_f", "axis2_i64"], ["is_bare_f3"]),
        helper.make_node("Mul", ["v_boost", "is_bare_f3"], ["boost_gated"]),
        helper.make_node("Mul", ["boost_gated", "onehot0"], ["boost_cls"]),
        helper.make_node("Add", ["v_value", "boost_cls"], ["gated_logits"]),
    ])
    del graph.output[:]
    graph.output.extend([
        helper.make_tensor_value_info("gated_cls", TensorProto.INT64, ["batch", "seq"]),
        helper.make_tensor_value_info(
            "gated_logits", TensorProto.FLOAT, ["batch", "seq", n_cls]),
    ])
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
        cls, logits = (o[0] for o in sess.run(["gated_cls", "gated_logits"], {"input": ids}))
        assert (logits.argmax(-1) == cls).all(), "argmax(gated_logits) != gated_cls"
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
