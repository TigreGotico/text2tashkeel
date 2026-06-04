# 4. Inference pipeline (string in, string out)

This page traces the **exact** path a sentence takes through the library, names
the tensor shapes, and explains two practical things you must know: **batching**
and the **INT8** model.

## 4.1 The five steps in code

For the default `bilstm` model, `Diacritizer.diacritize(text)` does this (see
`_BilstmBackend` in `text2tashkeel/_models.py`; `rawi` and `libtashkeel` follow
the same shape with their own vocab and decode):

```python
stripped = _strip(text)                       # ① normalize + remove any marks
ids = [[CHAR_TO_ID.get(c, UNK) for c in s]]   # ② letters → integer IDs
logits = session.run(["logits"], {"input_ids": ids})   # ③ run the ONNX model
pred = logits[0].argmax(-1)                    # ④ pick the top mark per letter
out  = "".join(ch + ID_TO_LABEL[c] for ch, c in zip(stripped, pred))  # ⑤ attach
return normalize(out)
```

### Step by step

| Step | What happens | Shape |
|------|--------------|-------|
| ① strip | NFKC-normalize, drop existing diacritics → bare consonants | `str` |
| ② encode | each character → its vocab ID (unknown → `<UNK>`=1) | `(1, N)` int64 |
| ③ run | the ONNX model produces a score for each of 15 marks per letter | `(1, N, 15)` float32 |
| ④ argmax | for each letter, the index of the highest score = predicted mark | `(N,)` |
| ⑤ apply | concatenate each base letter with its predicted mark, normalize | `str` |

`N` is the number of base characters in the input. Step ⑤ is exact because, by
construction, there is **one prediction per base character** — the alignment can
never drift.

> **Why the output can differ from the upstream demo.** The upstream project also
> consults a 30 MB sentence cache of known phrases before running the model. This
> library is **model-only** (to stay tiny), so on famous phrases its output is the
> model's best guess rather than a memorized gold answer. Same network, fewer
> crutches.

## 4.2 The ONNX contract

```
input  "input_ids" : int64   shape (batch, seq_len)
output "logits"    : float32 shape (batch, seq_len, 15)
```

Both `batch` and `seq_len` are **dynamic axes**, so any length works without
re-exporting. The model carries no tokenizer and no vocabulary — those live in
Python (`_CHAR_TO_ID`, `_ID_TO_LABEL`). That separation is deliberate: the heavy
math is in ONNX (portable to C++, Rust, the browser, mobile), while the trivial
text bookkeeping stays in whatever host language you use.

## 4.3 Batching — read this before you "speed it up" {#batching}

It is tempting to pad many sentences to a common length and run them as one
`(batch, seq_len)` tensor. **Don't.** The model's attention
([§3.5](03-architecture.md#35-bahdanau-attention--every-letter-looks-at-every-letter))
attends over **all** positions with **no padding mask**, so the `<PAD>` tokens
leak into every real position's context and change the predictions.

We verified this empirically: padding a short sentence to length 20/60/128
flipped the predicted mark on real characters in **every** sentence tested, and
shifted raw logits by as much as ~15. Concretely:

```
"هذا كتاب مفيد", unpadded   →  ...
same input, padded to 40    →  different marks on several letters
```

So the rule is simple:

- ✅ **Process one sentence per call.** `Diacritizer.diacritize` does exactly this.
- ✅ To go faster, run several **single-sentence** calls in parallel threads
  (onnxruntime releases the GIL during `run`), or across processes.
- ❌ Do **not** pad-and-stack variable-length sentences into one batch.

(If you control training, the fix is a masked attention; with these released
weights, per-sentence is the correct path. See
[`examples/02_batch.py`](../examples/02_batch.py) for the safe pattern.)

## 4.4 The INT8 model {#int8}

The default `bilstm` model ships in two precisions:

| Model name | File | Size | Use it when |
|------------|------|------|-------------|
| `bilstm` | `models/bilstm.onnx` | ~18 MB | default — best accuracy |
| `bilstm-int8` | `models/bilstm.int8.onnx` | ~4.5 MB | size matters more than accuracy |

`bilstm-int8` is produced by **dynamic INT8 quantization** (weights stored as
8-bit integers instead of 32-bit floats) — see
[`benchmarks/quantize.py`](../benchmarks/quantize.py). It is ~4× smaller and a bit
faster to load.

**The catch:** LSTMs are sensitive to quantization, so INT8 is **noticeably less
accurate** here — expect more wrong marks, and occasionally garbled output on
hard sentences. The exact cost is measured in
[`benchmarks/results.txt`](../benchmarks/results.txt) (DER for both models on the
same corpus). Use INT8 for size-constrained deployments (mobile, browser, edge)
where you've checked the accuracy drop is acceptable; otherwise stick with the
default fp32 model.

```python
from text2tashkeel import Diacritizer
small = Diacritizer("bilstm-int8")   # opt in explicitly
```

## 4.5 Metrics, defined {#metrics}

The benchmark reports two rates (lower is better), implemented verbatim from the
upstream `diacritize.evaluate`:

- **DER (Diacritic Error Rate)** — over all base characters, the fraction whose
  predicted mark differs from the gold mark.
  `DER = wrong_characters / total_characters`.
- **WER (Word Error Rate)** — over all whitespace-separated words, the fraction
  with **at least one** wrong character.
  `WER = wrong_words / total_words`.

Both compare *marks per base character* after NFKC normalization, with shadda
reordered to lead in compounds, so two visually identical strings always score as
equal. See [`benchmarks/benchmark.py`](../benchmarks/benchmark.py).

**Next:** [API reference →](05-api-reference.md)
