# text2tashkeel — documentation

A zero-to-hero guide to Arabic diacritization and the **lightweight, multi-model
toolkit** that does it. **No Arabic knowledge is assumed.** Every Arabic example is
romanized (written in Latin letters) and translated, so you can follow the
mechanics even if the script looks like decoration to you right now.

## What this library does, in one sentence

Arabic is normally written without its vowels. `text2tashkeel` reads vowel-less
Arabic and **puts the missing vowels back** — a **utility for lightweight
diacritization** that runs on `onnxruntime` (no PyTorch, no internet).

```
Input :  كتب محمد              ("ktb mhmd"   — bare consonants)
Output:  كَتَبَ مُحَمَّدٌ        ("kataba muhammadun" — "Muhammad wrote")
```

That task is called **diacritization**, or *tashkeel* (تشكيل) in Arabic — hence
the name `text2tashkeel`.

## A model picker, not a single model

The library is a **single tiny API over a set of interchangeable ONNX models** —
you choose the one that fits your accuracy / speed / size budget, and the
tokenization and decoding are handled for you. The set spans fast single models
(`rawi-v2`, `bilstm`, `libtashkeel`, and INT8 variants) and **gated ensembles**
that combine them for higher accuracy; the flagship default `rawi-ensemble` is one
such ensemble shipped as a single ONNX. Page
[8](08-models-and-benchmarks.md) compares accuracy *and speed*; page
[9](09-combining-models.md) explains the ensembles.

The deep-dive on model internals ([page 3](03-architecture.md)) walks through one
representative model end to end; the others share the overall shape but differ in
vocab, labels, and normalization.

## Read in this order

| # | Page | What you'll learn |
|---|------|-------------------|
| 1 | [Arabic script 101](01-arabic-script-101.md) | How the writing system works, what the vowel marks are, why they're usually left out. Start here even if you know zero Arabic. |
| 2 | [The diacritization problem](02-the-diacritization-problem.md) | Why "just add the vowels" is genuinely hard, and why it needs whole-sentence context. |
| 3 | [Architecture](03-architecture.md) | The neural network, layer by layer: tokenizer → embedding → BiLSTM → per-letter classifier (and the variants). |
| 4 | [Inference pipeline](04-inference-pipeline.md) | The exact path from a Python string to a diacritized string, including the ONNX tensor shapes and the batching gotcha. |
| 5 | [API reference](05-api-reference.md) | Every public function and class. |
| 6 | [Glossary](06-glossary.md) | Quick definitions of every Arabic and ML term used here. |
| 7 | [Credits & license](07-credits-and-license.md) | Where each model comes from. **Full credit to the original authors.** |
| 8 | [Models & benchmarks](08-models-and-benchmarks.md) | The bundled models compared — and why the scores need a big asterisk. |
| 9 | [Combining models](09-combining-models.md) | How gated ensembles split *where* a mark goes from *which* mark it is — and how the flagship combines two models with no retraining. |
| 10 | [Full benchmark report](10-benchmark-report.md) | All models on the full test set — accuracy vs **latency** vs size, with Pareto plots. The voice-pipeline picker. |
| 11 | [What makes rawi different](11-what-makes-rawi-different.md) | rawi's distinctive features — a wider task (restores hamza/dagger-alef) and a where/which factorization, not a fancy network. |
| 12 | [Alternative architectures](12-alternative-architectures.md) | How rawi compares to other diacritizers (attention RNNs, transformers, rule engines) within the **embeddable tier** (≤ 25 MB, ≤ 20 ms) — and why it's the inference shape, not the parameter count, that decides speed. |

## Try it in 30 seconds

```python
from text2tashkeel import diacritize
print(diacritize("بسم الله الرحمن الرحيم"))
# بِسمِ اللَّهِ الرَّحمَنِ الرَّحِيمِ   ("bismi-llāhi-r-raḥmāni-r-raḥīm")
```

Then open [`examples/`](../examples/README.md) for commented, runnable scripts
that print the intermediate steps so you can *see* the model thinking.

## Credit where it's due

The **rawi** family — `rawi` (V1), `rawi-v2`, the two-head `rawi-v3`, and the
`rawi-ensemble` flagship — is original work by **TigreGotico**: the models, the
corpus, the gated-ensemble method, and the analysis behind them (Mike Hansen ran the
V2/V3 training). Two further models are independent third-party work, re-exported
here with credit: **`bilstm`** (a BiLSTM + Bahdanau-attention diacritizer) by **Zain
Mahmood**, and **`libtashkeel`** by **Musharraf Omer** — both MIT. See
[Credits & license](07-credits-and-license.md) for the full breakdown.
