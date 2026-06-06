# 5. API reference

The public surface is intentionally tiny: one class, two functions, and a CLI.
Everything under `text2tashkeel._models` (leading underscore) is private and may
change.

## `text2tashkeel.available_models() -> list[str]`

The model names you can pass to `Diacritizer`:

```python
>>> available_models()
['bilstm', 'bilstm-int8', 'rawi', 'rawi-int8', 'rawi-v2', 'rawi-v2-int8',
 'rawi-v3', 'rawi-v3-int8', 'libtashkeel',
 'bilstm+rawi', 'bilstm+rawi-int8', 'libtashkeel+rawi',
 'libtashkeel+rawi-int8', 'bilstm-int8+rawi-int8', 'bilstm+libtashkeel+rawi',
 'bilstm+libtashkeel+rawi-int8', 'rawi-v2+rawi', 'rawi-v2+rawi-int8',
 'rawi-v2-int8+rawi-int8',
 'rawi-v2+rawi-v3', 'rawi-v2-int8+rawi-v3-int8',
 'rawi-ensemble']
```

Gated-ensemble names spell out the combo: **`gate(+gate)+value`** — the last model
is the *value* (decides *which* mark), the rest are *gates* (decide *where*,
OR-combined). See [Models & benchmarks](08-models-and-benchmarks.md) for what each
is and the [full report](10-benchmark-report.md) for accuracy/latency/size; the
gating story is in [Combining models](09-combining-models.md).

`available_models(bundled_only=True)` returns only the models whose weights ship in
the wheel (the INT8 singles + the stitched flagship) — these run offline with no
download. The fp32 variants are fetched from Hugging Face on first use when the
`hf` extra is installed (`pip install text2tashkeel[hf]`); without it, requesting one
raises a `FileNotFoundError` naming the model's HF repo.

## `text2tashkeel.register_model(name, onnx_path, vocab_path=None, *, arch="rawi", threshold=0.5)`

Register a model from your own files — e.g. one trained on a different corpus — so
`Diacritizer(name)` can use it.

| Parameter | Notes |
|-----------|-------|
| `name` | the model name to register (reusing a name overrides it) |
| `onnx_path` | your exported ONNX |
| `vocab_path` | matching vocab JSON (`char_to_idx`/`diac_to_idx` for `rawi`/`rawi-v3`/`stitched`; maps JSON for `libtashkeel`; omit for `bilstm`) |
| `arch` | decode to use: `"rawi"` (single head), `"rawi-v3"`/`"two-head"`, `"stitched"`, `"bilstm"`, `"libtashkeel"` |
| `threshold` | presence threshold for two-head models (default 0.5) |

```python
from text2tashkeel import register_model, Diacritizer
register_model("my-rawi", "my_model.onnx", "my_vocab.json", arch="rawi")
Diacritizer("my-rawi").diacritize("نص عربي")
```

## Per-model wrapper classes (syntactic sugar)

One thin `Diacritizer` subclass per model, so you can name the model directly
(and get autocomplete) instead of passing a string:

| Class | model | Class | model |
|-------|-------|-------|-------|
| `Bilstm()` | `bilstm` | `BilstmRawi()` | `bilstm+rawi` |
| `BilstmInt8()` | `bilstm-int8` | `BilstmRawiInt8()` | `bilstm+rawi-int8` |
| `Rawi()` | `rawi` | `LibtashkeelRawi()` | `libtashkeel+rawi` |
| `RawiInt8()` | `rawi-int8` | `LibtashkeelRawiInt8()` | `libtashkeel+rawi-int8` |
| `RawiV3()` | `rawi-v3` (two-head gated) | `RawiV3Int8()` | `rawi-v3-int8` |
| `RawiV2()` | `rawi-v2` | `BilstmInt8RawiInt8()` | `bilstm-int8+rawi-int8` |
| `RawiV2Int8()` | `rawi-v2-int8` (lean single) | `BilstmLibtashkeelRawi()` | `bilstm+libtashkeel+rawi` |
| `Libtashkeel()` | `libtashkeel` | `BilstmLibtashkeelRawiInt8()` | `bilstm+libtashkeel+rawi-int8` |
| `RawiV2Rawi()` | `rawi-v2+rawi` (most accurate) | `RawiV2RawiInt8()` | `rawi-v2+rawi-int8` |
| `RawiV2Int8RawiInt8()` | `rawi-v2-int8+rawi-int8` (fully int8) | `RawiV2RawiV3()` | `rawi-v2+rawi-v3` (flagship) |
| `RawiV2Int8RawiV3Int8()` | `rawi-v2-int8+rawi-v3-int8` | `RawiEnsemble()` ⭐ | `rawi-ensemble` (default, stitched) |

```python
from text2tashkeel import RawiEnsemble, RawiV2Int8, Bilstm
RawiEnsemble().diacritize("بسم الله الرحمن الرحيم")   # flagship default (2.04% DER)
RawiV2Int8()("هذا كتاب مفيد")                          # lean single model (~1 ms, 2.5 MB)
```

Each accepts the same `providers=` argument and behaves identically to the
string form. They're pure sugar over the model registry below.

## `text2tashkeel.Diacritizer`

```python
Diacritizer(model="rawi-ensemble", providers=["CPUExecutionProvider"])
```

A reusable diacritizer holding one onnxruntime session.

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `model` | `str` | `"rawi-ensemble"` | a name from `available_models()` |
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
pass already-diacritized text (it will be re-diacritized). Returns NFC-normalized
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
logits, custom decoding). Each model has a different ONNX I/O signature (rawi:
`input → output`; rawi-ensemble: `input → gated_cls`; `bilstm`:
`input_ids → logits`; `libtashkeel`: three inputs); see
[the inference page](04-inference-pipeline.md#42-the-onnx-contract) and
`examples/06_raw_onnx.py`.

```python
d = Diacritizer("bilstm")
sess = d.backend.sess
import numpy as np
ids = np.array([[14, 9, 11]], dtype=np.int64)               # (batch, seq_len)
logits = sess.run(["logits"], {"input_ids": ids})[0]        # (batch, seq_len, 15)
```

## `text2tashkeel.diacritize(text, model="rawi-ensemble") -> str`

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
