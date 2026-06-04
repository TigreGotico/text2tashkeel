# text2tashkeel

Tiny, standalone Arabic diacritizer (tashkeel) — it puts the missing vowel marks
back into Arabic text. Several diacritization models, all running on
`onnxruntime`: **no PyTorch, no network, no API keys.** Every model ships inside
the package; the only runtime dependencies are `numpy` and `onnxruntime`.

```python
from text2tashkeel import Diacritizer
Diacritizer().diacritize("بسم الله الرحمن الرحيم")
# 'بِسمِ اللَّهِ الرَّحمَنِ الرَّحِيمِ'
```

New to Arabic or diacritization? The [`docs/`](docs/index.md) are a **zero-to-hero
guide written for non-Arabic speakers** — the writing system, the linguistics, the
model architecture, a glossary. Start at [`docs/index.md`](docs/index.md).

## Install

```bash
pip install -e .
```

## Models

Interchangeable, self-contained models from three independent projects, plus two
gated ensembles built from them:

| Name | Size | Origin |
|------|------|--------|
| `bilstm` (default) | ~18 MB | [Z-Mahmood](https://github.com/Z-Mahmood/arabic-diacritizer-public-release) — BiLSTM + attention |
| `bilstm-int8` | ~4.5 MB | INT8 quantization (smaller, less accurate) |
| `rawi` | ~9.8 MB | [TigreGotico/rawi](https://huggingface.co/TigreGotico/rawi) |
| `libtashkeel` | ~4.8 MB | [mush42/libtashkeel](https://github.com/mush42/libtashkeel) |
| `bilstm+rawi` | — | gated ensemble (2 models) |
| `ensemble` | — | gated ensemble (3 models) — **most accurate** |

```python
from text2tashkeel import Diacritizer, available_models
available_models()
# ['bilstm', 'bilstm-int8', 'rawi', 'libtashkeel', 'bilstm+rawi', 'ensemble']
Diacritizer("ensemble").diacritize("بسم الله الرحمن الرحيم")   # best quality
```

The **`ensemble`** combines the models so that one decides *where* marks go and
another decides *which* mark — beating every standalone model with no retraining.
The story is in [`docs/09-combining-models.md`](docs/09-combining-models.md).

`Diacritizer` is callable (`d("...")`) and lazily builds **one** onnxruntime
session it reuses — construct once, call many times. Full credits and licenses for
every model: [`docs/07-credits-and-license.md`](docs/07-credits-and-license.md).

## CLI

```bash
text2tashkeel "الحمد لله رب العالمين"
echo "محمد رسول الله" | text2tashkeel
text2tashkeel -m libtashkeel < input.txt > output.txt
```

## How it works (in one paragraph)

Text is normalized and stripped of any existing marks, mapped to character IDs,
and fed to a small neural net that predicts a diacritic class **per base
character**; the predicted marks are re-applied. The heavy math is in ONNX
(`input_ids → logits`); tokenization and mark-application are pure Python. Both
tensor axes are dynamic, so any length works — but **process one sentence per
call**: the models' attention is unmasked, so padding to batch changes the
answers (details + proof in
[`docs/04-inference-pipeline.md`](docs/04-inference-pipeline.md#batching)).

## Benchmarks

Measured DER/WER for every model across the corpus's train/test/val splits is in
[`benchmarks/`](benchmarks/README.md). **Important:** the corpus overlaps every
model's training data, so the numbers are indicative, not a clean ranking — see
[`docs/08-models-and-benchmarks.md`](docs/08-models-and-benchmarks.md#82--the-contamination-caveat--read-this-before-the-numbers).

## Scope & credits

Model-only path — upstream sentence caches/hint features are omitted to keep this
small and uniform. Targets Modern Standard / classical Arabic; dialect is
untested. This package is a **repackaging** (ONNX exports + a unified pure-Python
wrapper + docs + benchmarks); all model weights and architectures are the work of
their original authors — see [credits](docs/07-credits-and-license.md). MIT for
the wrapper; bundled models keep their upstream licenses.
