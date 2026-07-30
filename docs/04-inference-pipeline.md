# 4. Inference pipeline (string in, string out)

This page traces the exact path a sentence takes through the library, names
the tensor shapes, and explains two practical points: batching and INT8.

## 4.1 The five steps in code

`Diacritizer.diacritize(text)` normalizes the text, runs the model, and
re-applies the predicted marks. For a single rawi model (`_RawiBackend` in
`text2tashkeel/_models.py`):

```python
bare = "".join(c for c in nfd_drop_symbols(text)      # ① NFD, drop symbols + marks
               if category(c) != "Mn")
ids  = [[char_to_idx.get(c, UNK) for c in bare]]      # ② letters → integer IDs
cls  = session.run(["output"], {"input": ids})[0][0].argmax(-1)   # ③ run ONNX
out  = "".join(ch + (idx_to_diac[c] if is_letter(ch) else "")     # ④ attach marks
               for ch, c in zip(bare, cls))
return nfc(out)                                       # ⑤ recompose to NFC
```

| Step | What happens | Shape |
|------|--------------|-------|
| ① normalize | NFD, drop symbol chars (`So`) and existing marks (`Mn`) → bare base letters | `str` |
| ② encode | each character → its vocab ID (unknown → `<UNK>`=1) | `(1, N)` int64 |
| ③ run | the ONNX model scores each diacritic class per letter | `(1, N, C)` float32 |
| ④ argmax + attach | top class per letter → its mark, attached to the base letter | — |
| ⑤ recompose | NFC-normalize the result | `str` |

`N` is the number of base characters, and `C` is the class count (73/75 for
rawi, 15 for bilstm). There is one prediction per base character, so the
alignment never drifts. The flagship `rawi-ensemble` follows the same shape
with one extra wrinkle: its ONNX returns the already-gated class directly
(output `gated_cls`), so step 3 is one `session.run` over the fused graph.

## 4.2 The ONNX contract

Each model carries only the network. Tokenization and mark-application stay
in Python. The signatures differ slightly:

| Model | input | output |
|-------|-------|--------|
| rawi (V1/V2) | `input` int64 `(batch, seq)` | `output` float32 `(batch, seq, C)` |
| rawi-v3 | `input` | `presence (batch, seq)`, `value (batch, seq, 75)` |
| rawi-ensemble | `input` | `gated_cls` int64 `(batch, seq)` |
| bilstm | `input_ids` | `logits` float32 `(batch, seq, 15)` |

Both axes are dynamic, so any length works without re-exporting. Keeping the
math in ONNX, which is portable to C++, Rust, browsers, and mobile, and
keeping the text bookkeeping in the host language, is a deliberate design
choice.

## 4.3 Batching: read this before you try to speed it up {#batching}

It is tempting to pad many sentences to a common length and run them as one
`(batch, seq)` tensor. Do not do this. The encoders are unmasked. The
bidirectional LSTM's reverse pass reads the padding before it reaches the
real letters, and `bilstm` also attends over all positions with no
padding mask. So `<PAD>` tokens leak into real positions' context and
change the predictions.

The rule:

- Process one sentence per call. `Diacritizer.diacritize` does exactly this.
- To go faster, run several single-sentence calls in parallel threads
  (onnxruntime releases the GIL during `run`), or across processes.
- Do not pad and stack variable-length sentences into one batch.

See [`examples/05_batch_safely.py`](../examples/05_batch_safely.py) for the
safe pattern.

## 4.4 INT8, and why the default is INT8 {#int8}

Every bundled model is INT8, and so is the default `rawi-ensemble`. INT8
stores weights as 8-bit integers instead of 32-bit floats, about 4 times
smaller and a bit faster to load, produced by dynamic quantization
([`benchmarks/quantize.py`](../benchmarks/quantize.py)).

Whether INT8 costs accuracy depends on the architecture:

| Model | fp32 → INT8 DER | verdict |
|-------|-----------------|---------|
| rawi / rawi-v2 / rawi-v3 (no attention) | about no change (for example 2.29% → about 2.30%) | lossless |
| bilstm (BiLSTM + attention) | 4.95% → 12.89% | lossy — attention matmuls are quant-sensitive |

So the flagship and the rawi singles ship and run as INT8 with no accuracy
penalty. That is why the wheel can be small and accurate at the same time.
Only `bilstm` pays a cost for INT8. Its fp32 form is the one to use for that
model. See [§8.4](08-models-and-benchmarks.md#quantization-is-architecture-dependent)
for details and the [benchmark report](10-benchmark-report.md) for full
per-model numbers.

```python
from text2tashkeel import Diacritizer
Diacritizer()                  # rawi-ensemble — INT8, 2.04% DER, the default
Diacritizer("rawi-v2-int8")    # lean single INT8 model
```

## 4.5 Metrics, defined {#metrics}

The benchmarks report (lower is better):

- DER (Diacritic Error Rate): over all base characters, the fraction whose
  predicted mark differs from the gold mark.
- DER\*: the same, restricted to characters that carry a mark in the gold
  text. This is stricter; see
  [§8.3](08-models-and-benchmarks.md#83-two-error-rates-because-der-is-ambiguous).
- WER (Word Error Rate): the fraction of whitespace-separated words with at
  least one wrong mark.

All these compare marks per base character after NFC normalization, with
shadda reordered to lead in compounds, so two visually identical strings
score as equal. See [`benchmarks/benchmark.py`](../benchmarks/benchmark.py).

---
[← Architecture](03-architecture.md) · [Home](index.md) · [Next →](05-api-reference.md)
