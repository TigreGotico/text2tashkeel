# text2tashkeel

A **utility for lightweight Arabic diacritization** (tashkeel) — it puts the
missing vowel marks back into Arabic text. Not one model but a **model picker**: a
single tiny API over interchangeable diacritization models, all running on
`onnxruntime` — **no PyTorch, no API keys, offline by default.** Pick the model 
that fits your accuracy/speed/size budget; the only runtime
dependencies are `numpy` and `onnxruntime`.

```python
from text2tashkeel import Diacritizer
Diacritizer().diacritize("بسم الله الرحمن الرحيم")              # default model - 2.04% DER
Diacritizer("rawi-v2-int8").diacritize("بسم الله الرحمن الرحيم")  # lean single model
```

> **More than vowels.** Most diacritizers only add the short-vowel marks to text
> that is *already spelled correctly*. The default rawi models also **restore the
> hamza (ء) and the silent dagger-alef** — so they fix real, inconsistently-spelled
> input (e.g. a bare `ا` typed for `أ`), not just clean text. This is rare among
> diacritizers; [here's exactly why and how](docs/11-what-makes-rawi-different.md#111-a-wider-task-rawi-restores-hamza-and-the-dagger-alef-not-just-vowels).

## Install

```bash
pip install text2tashkeel
```

The wheel is small (~10 MB): it bundles our best models which work fully offline (no downloads, no torch). 
The full-precision (fp32) variants are fetched from Hugging Face **on first use** if you opt in:

```bash
pip install text2tashkeel        # int8 + flagship, offline
pip install text2tashkeel[hf]    # + auto-download fp32 models on demand
```

Without `[hf]`, asking for a non-bundled model raises a clear message with its
Hugging Face link. You can also point at **your own model** (e.g. one trained on a
different corpus) with `register_model(...)` — see below. For development:
`pip install -e ".[test]"` then `pytest`.

## Models

Two models cover almost every use; both ship in the wheel and run offline:

| Use case | Model | DER ↓ | latency | size |
|----------|-------|------:|--------:|-----:|
| **best accuracy (default)** ⭐ | `rawi-ensemble` | **2.04%** | ~2 ms | 4.9 MB |
| **fastest & smallest** | `rawi-v2-int8` | 2.30% | **~1 ms** | **2.5 MB** |

**22 model configurations** are available — the rawi family (V1/V2/V3 + INT8), two
independent diacritizers (`bilstm` and `libtashkeel`), and gated ensembles of them —
for comparison, research, or special cases:

```python
from text2tashkeel import available_models, Diacritizer
available_models()                 # all models
available_models(bundled_only=True)  # the models that ship in the wheel (offline)
Diacritizer("rawi-v2-int8").diacritize("بسم الله الرحمن الرحيم")
```

**Bundled vs fetched.** `available_models(bundled_only=True)` lists the models that
ship in the wheel. Everything else downloads from Hugging Face on first use with `[hf]` installed;
each model's weights live in its own repo
([`rawi`](https://huggingface.co/TigreGotico/rawi),
[`rawi-v2`](https://huggingface.co/TigreGotico/rawi-v2),
[`rawi-v3`](https://huggingface.co/TigreGotico/rawi-v3),
[`rawi-ensemble`](https://huggingface.co/TigreGotico/rawi-ensemble),
[`bilstm`](https://huggingface.co/TigreGotico/bilstm-diacritizer),
[`libtashkeel`](https://huggingface.co/TigreGotico/libtashkeel-diacritizer)), all
grouped in the [**Arabic Diacritizers** collection](https://huggingface.co/collections/TigreGotico/arabic-diacritizers-tashkeel-6a247318559bcc49e128aa5f).

**Bring your own model.** Trained a diacritizer on a different corpus? Point at it:

```python
from text2tashkeel import register_model, Diacritizer
register_model("my-rawi", "my_model.onnx", "my_vocab.json", arch="rawi")  # or arch="rawi-v3"
Diacritizer("my-rawi").diacritize("نص عربي")
```

`Diacritizer` is callable (`d("...")`) and lazily builds **one** onnxruntime
session it reuses — construct once, call many times. Full credits and licenses for
every model: [`docs/07-credits-and-license.md`](docs/07-credits-and-license.md).

## CLI

```bash
text2tashkeel "الحمد لله رب العالمين"          # flagship default
echo "محمد رسول الله" | text2tashkeel
text2tashkeel -m rawi-v2-int8 < input.txt > output.txt
```

## Benchmarks

Measured DER/WER for every model across the corpus's train/test/val splits is in [`benchmarks/`](benchmarks/README.md). 
