# text2tashkeel — documentation

A zero-to-hero guide to Arabic diacritization and the tiny model that powers
this library. **No Arabic knowledge is assumed.** Every Arabic example is
romanized (written in Latin letters) and translated, so you can follow the
mechanics even if the script looks like decoration to you right now.

## What this library does, in one sentence

Arabic is normally written without its vowels. This library reads vowel-less
Arabic and **puts the missing vowels back**, using a small neural network that
runs on `onnxruntime` (no PyTorch, no internet).

```
Input :  كتب محمد              ("ktb mhmd"   — bare consonants)
Output:  كَتَبَ مُحَمَّدٌ        ("kataba muhammadun" — "Muhammad wrote")
```

That task is called **diacritization**, or *tashkeel* (تشكيل) in Arabic — hence
the name `text2tashkeel`.

The package bundles **several diacritization models** from independent projects
(`bilstm`, `rawi`, `libtashkeel`, plus a quantized variant), all behind one tiny
API. Page [8](08-models-and-benchmarks.md) compares them.

## Read in this order

| # | Page | What you'll learn |
|---|------|-------------------|
| 1 | [Arabic script 101](01-arabic-script-101.md) | How the writing system works, what the vowel marks are, why they're usually left out. Start here even if you know zero Arabic. |
| 2 | [The diacritization problem](02-the-diacritization-problem.md) | Why "just add the vowels" is genuinely hard, and why it needs whole-sentence context. |
| 3 | [Architecture](03-architecture.md) | The neural network, layer by layer: tokenizer → embedding → BiLSTM → attention → classifier. |
| 4 | [Inference pipeline](04-inference-pipeline.md) | The exact path from a Python string to a diacritized string, including the ONNX tensor shapes and the batching gotcha. |
| 5 | [API reference](05-api-reference.md) | Every public function and class. |
| 6 | [Glossary](06-glossary.md) | Quick definitions of every Arabic and ML term used here. |
| 7 | [Credits & license](07-credits-and-license.md) | Where each model comes from. **Full credit to the original authors.** |
| 8 | [Models & benchmarks](08-models-and-benchmarks.md) | The bundled models compared — and why the scores need a big asterisk. |
| 9 | [Combining models](09-combining-models.md) | A research arc: how gating two models' decisions produces the most accurate diacritizer here, with no retraining. |

## Try it in 30 seconds

```python
from text2tashkeel import diacritize
print(diacritize("بسم الله الرحمن الرحيم"))
# بِسمِ اللَّهِ الرَّحمَنِ الرَّحِيمِ   ("bismi-llāhi-r-raḥmāni-r-raḥīm")
```

Then open [`examples/`](../examples/README.md) for commented, runnable scripts
that print the intermediate steps so you can *see* the model thinking.

## Credit where it's due

The model weights and the BiLSTM-plus-attention architecture are the work of
**Zain Mahmood**, released as
[`Z-Mahmood/arabic-diacritizer-public-release`](https://github.com/Z-Mahmood/arabic-diacritizer-public-release)
under the MIT license. This library is a thin, dependency-light repackaging:
the PyTorch model exported to ONNX, wrapped in a minimal pure-Python interface.
See [Credits & license](07-credits-and-license.md) for the full story of what is
original and what is repackaged.
