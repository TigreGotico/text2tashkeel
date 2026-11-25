# Install, models and CLI

## Install

```bash
pip install --pre text2tashkeel
```

The `--pre` flag installs the latest alpha. The stable release on PyPI is
older and lacks the orthography constraints and `waqf`.

The wheel is small (about 10 MB). It bundles the best models, and these work
fully offline, with no downloads and no torch. The full-precision (fp32)
variants are fetched from Hugging Face on first use, if you opt in:

```bash
pip install --pre text2tashkeel        # int8 + flagship, offline
pip install --pre "text2tashkeel[hf]"  # + auto-download fp32 models on demand
```

Without `[hf]`, asking for a non-bundled model raises a clear message with its
Hugging Face link. You can also point at your own model, such as one trained
on a different corpus, with `register_model(...)`. See [Models](#models). For
development, run `pip install -e ".[test]"`, then run `pytest`.

## Models

Two models cover almost every use. Both ship in the wheel and run offline.

| Use case | Model | DER ↓ | latency | size |
|----------|-------|------:|--------:|-----:|
| **best accuracy (default)** ⭐ | `rawi-ensemble` | **2.03%** | ~2 ms | 4.9 MB |
| **fastest & smallest** | `rawi-v2-int8` | 2.30% | **~1 ms** | **2.5 MB** |

22 model configurations are available for comparison, research, or special
cases: the rawi family (V1/V2/V3 + INT8), two independent diacritizers
(`bilstm` and `libtashkeel`), and gated ensembles of them.

```python
from text2tashkeel import available_models, Diacritizer
available_models()                 # all models
available_models(bundled_only=True)  # the models that ship in the wheel (offline)
Diacritizer("rawi-v2-int8").diacritize("بسم الله الرحمن الرحيم")
```

`available_models(bundled_only=True)` lists the models that ship in the
wheel. Everything else downloads from Hugging Face on first use, with `[hf]`
installed. Each model's weights live in its own repo
([`rawi`](https://huggingface.co/TigreGotico/rawi),
[`rawi-v2`](https://huggingface.co/TigreGotico/rawi-v2),
[`rawi-v3`](https://huggingface.co/TigreGotico/rawi-v3),
[`rawi-ensemble`](https://huggingface.co/TigreGotico/rawi-ensemble),
[`bilstm`](https://huggingface.co/TigreGotico/bilstm-diacritizer),
[`libtashkeel`](https://huggingface.co/TigreGotico/libtashkeel-diacritizer)),
grouped in the [Arabic Diacritizers collection](https://huggingface.co/collections/TigreGotico/arabic-diacritizers-tashkeel-6a247318559bcc49e128aa5f).

If you trained a diacritizer on a different corpus, point at it:

```python
from text2tashkeel import register_model, Diacritizer
register_model("my-rawi", "my_model.onnx", "my_vocab.json", arch="rawi")  # or arch="rawi-v3"
Diacritizer("my-rawi").diacritize("نص عربي")
```

`Diacritizer` is callable (`d("...")`) and lazily builds one onnxruntime
session, which it reuses. Construct it once, then call it many times. For
full credits and licenses for every model, see
[`docs/07-credits-and-license.md`](07-credits-and-license.md).

## CLI

```bash
text2tashkeel "الحمد لله رب العالمين"          # flagship default
echo "محمد رسول الله" | text2tashkeel
text2tashkeel -m rawi-v2-int8 < input.txt > output.txt
```


---

[← What makes rawi different](11-what-makes-rawi-different.md) · [Home](index.md)
