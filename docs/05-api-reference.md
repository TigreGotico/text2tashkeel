# 5. API reference

The public surface is intentionally tiny: one class, two functions, and a CLI.
Everything under `text2tashkeel._models` (leading underscore) is private and may
change.

## `text2tashkeel.available_models() -> list[str]`

The model names you can pass to `Diacritizer`:

```python
>>> available_models()
['bilstm', 'bilstm-int8', 'rawi', 'libtashkeel', 'bilstm+rawi', 'ensemble']
```

See [Models & benchmarks](08-models-and-benchmarks.md) for what each one is, and
[Combining models](09-combining-models.md) for the `ensemble` / `bilstm+rawi`
gated combinations (the most accurate option is `ensemble`).

## `text2tashkeel.Diacritizer`

```python
Diacritizer(model="bilstm", providers=["CPUExecutionProvider"])
```

A reusable diacritizer holding one onnxruntime session.

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `model` | `str` | `"bilstm"` | a name from `available_models()` |
| `providers` | `list[str]` | `["CPUExecutionProvider"]` | e.g. `["CUDAExecutionProvider", "CPUExecutionProvider"]` if you have GPU ORT |

A bad `model` name raises `ValueError`. The onnxruntime session is built
**lazily** on first use, so constructing a `Diacritizer` is cheap; the cost is
paid on the first `diacritize` call.

```python
Diacritizer("libtashkeel")        # pick a model
Diacritizer("bilstm-int8")        # the small quantized model
```

### `.diacritize(text: str) -> str`

Diacritize one sentence/string. Strips any existing marks first, so it's safe to
pass already-diacritized text (it will be re-diacritized). Returns NFKC-normalized
output. An empty/whitespace-only input returns itself.

```python
from text2tashkeel import Diacritizer
d = Diacritizer()
d.diacritize("نص عربي")     # 'نَصٌ عَرَبِيٌّ'
d("نص عربي")                # __call__ is an alias for .diacritize
```

Process **one sentence per call** — do not pad-and-batch (see
[batching](04-inference-pipeline.md#batching)). For throughput, run multiple calls
across threads; onnxruntime releases the GIL during inference.

### `.backend`

The internal backend object for the chosen model. Its `.sess` attribute is the
underlying `onnxruntime.InferenceSession` — use it for advanced needs (raw
logits, custom decoding). Note each model has a different ONNX I/O signature
(`bilstm`: `input_ids → logits`; `libtashkeel`: three inputs); see
[the inference page](04-inference-pipeline.md) and `examples/06_raw_onnx.py`.

```python
d = Diacritizer("bilstm")
sess = d.backend.sess
import numpy as np
ids = np.array([[14, 9, 11]], dtype=np.int64)               # (batch, seq_len)
logits = sess.run(["logits"], {"input_ids": ids})[0]        # (batch, seq_len, 15)
```

## `text2tashkeel.diacritize(text, model="bilstm") -> str`

Module-level convenience using a shared, lazily-created default `Diacritizer` per
model name. Ideal for quick scripts.

```python
from text2tashkeel import diacritize
diacritize("هذا كتاب مفيد")
diacritize("هذا كتاب مفيد", model="libtashkeel")
```

For repeated use in long-running services, prefer constructing your own
`Diacritizer` so you control the model and providers.

## CLI

Installed as the `text2tashkeel` command (and runnable as `python -m
text2tashkeel`).

```bash
text2tashkeel "الحمد لله رب العالمين"     # argument
echo "محمد رسول الله" | text2tashkeel      # stdin, line by line
text2tashkeel < input.txt > output.txt     # file via redirection
```

It diacritizes each input line and prints the result; blank lines pass through.

## Internals (private, for the curious)

In `text2tashkeel._models`:

| Name | What it is |
|------|------------|
| `_strip(text)` | NFC-normalize and remove the tashkeel marks |
| `_BILSTM_C2I` | bilstm's 54-symbol vocabulary (must match the trained tokenizer) |
| `_BILSTM_ID2LABEL` | bilstm's class index 0–14 → diacritic string |
| `_BilstmBackend` / `_RawiBackend` / `_LibtashkeelBackend` | the per-model encode/decode logic |
| `build_backend(name)` | construct (and cache) a backend by name |

These vocab tables are covered by `tests/test_vocab.py` precisely because a
mismatch would silently corrupt every prediction. The ports are pinned to their
upstream reference outputs by `tests/test_upstream_parity.py`.

**Next:** [Glossary →](06-glossary.md)
