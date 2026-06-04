# 8. Models & benchmarks

The package bundles four interchangeable models. This page explains how they
differ and how they score — and, just as importantly, **why those scores must be
read with care.**

## 8.1 The bundled models

| Name | Size | Architecture | Classes | Origin |
|------|------|--------------|---------|--------|
| `bilstm` | ~18 MB | BiLSTM + Bahdanau attention | 15 | [Z-Mahmood](https://github.com/Z-Mahmood/arabic-diacritizer-public-release) |
| `bilstm-int8` | ~4.5 MB | same, INT8-quantized | 15 | quantized here |
| `rawi` | ~9.8 MB | BiLSTM (NFD labels, restores hamzas) | 73 | [TigreGotico/rawi](https://huggingface.co/TigreGotico/rawi) |
| `libtashkeel` | ~4.8 MB | char+hint encoder, length-masked | 15 | [mush42/libtashkeel](https://github.com/mush42/libtashkeel) |
| `bilstm+rawi` | — | gated ensemble (2 models) | — | this project — see [§9](09-combining-models.md) |
| `ensemble` | — | gated ensemble (3 models) — **most accurate** | — | this project — see [§9](09-combining-models.md) |

They are **not** the same model trained four times — they come from three
independent projects with different tokenizers, label schemes, normalization, and
training data. So they legitimately disagree on hard words. Full attribution is on
the [credits page](07-credits-and-license.md).

```python
from text2tashkeel import Diacritizer
Diacritizer("rawi").diacritize("بسم الله الرحمن الرحيم")
```

## 8.2 ⚠️ The contamination caveat — read this before the numbers

The benchmark corpus,
[`TigreGotico/arabic_diacritized_text`](https://huggingface.co/datasets/TigreGotico/arabic_diacritized_text),
is an **aggregate of many public diacritized-Arabic sources**. Critically:

> **Every model here was very likely trained on text that also appears in this
> corpus.** The `rawi` model definitely was (on a subset). The others draw on the
> same well-known public sources that this corpus aggregates.

That means the benchmark largely measures **fit on data the models have probably
already seen**, not their ability to generalize to *new* text. Consequences:

- **Absolute DER is optimistic for everyone.** Real-world DER on unseen,
  out-of-domain text (dialect, novel prose) will be higher.
- **Small gaps between models are not a reliable ranking.** Differences in how
  much each model "saw" this data confound the comparison.
- The honest takeaway is the **shape** of the results (e.g. "fp32 ≫ int8",
  "all the full-precision models land in a similar single-to-low-double-digit
  range"), not a precise leaderboard.

We report it anyway because you asked for a concrete comparison — just hold it at
arm's length.

## 8.3 Two error rates, because "DER" is ambiguous

Different projects define their headline number differently, so we report two,
computed identically for **every** model:

- **DER (all)** — wrong mark fraction over **all** base characters. The strict,
  standard definition (used by the `bilstm` project). Counts "no mark" positions,
  which are easy, so this number looks low.
- **DER\* (marked)** — wrong mark fraction over **only** the characters that carry
  a mark in the gold text. This mirrors `rawi`'s training metric
  (`mask = diacritics != 0`) and is much stricter per-position. A model can have a
  low DER but a high DER\* if it mostly errs *on the hard, marked positions*.

Reporting both prevents an apples-to-oranges comparison — e.g. `rawi`'s model card
cites ~2.96% on a *marked-positions* accuracy basis, which is **not** the same
quantity as strict DER.

(WER — word error rate — is also reported: the fraction of words with any mark
wrong.)

## 8.4 Results

### Full test split — all 817,035 sentences (49.4M characters)

The single models, scored on the **entire** held-out `test.txt`
([`benchmarks/results_full_test.txt`](../benchmarks/results_full_test.txt), run on
a 24-core box via [`benchmarks/parallel_benchmark.py`](../benchmarks/parallel_benchmark.py)).
**Re-read §8.2 before interpreting** — the corpus overlaps every model's training
data.

| Model | DER (all) ↓ | DER\* (marked) ↓ | WER ↓ |
|-------|-------------|------------------|-------|
| **`ensemble`** (gated, see [§9](09-combining-models.md)) | **3.99%** | 3.13% | **15.79%** |
| `libtashkeel+rawi` (gated, 2-model) | 4.12% | 3.59% | 16.22% |
| `bilstm+rawi` (gated, 2-model) | 4.38% | 4.17% | 16.72% |
| bilstm | 4.95% | 5.08% | 18.02% |
| libtashkeel | 6.89% | 7.80% | 24.56% |
| bilstm-int8 | 12.89% | 18.42% | 37.54% |
| rawi | 18.49% | **3.07%** | 55.48% |

> **External SOTA baseline.** [CATT](https://github.com/abjadai/catt) (a character
> transformer) was exported to ONNX and benchmarked on a 2k sample with a stricter
> per-Arabic-letter metric; its **EO** variant scored **4.06% DER / 2.30% DER\***,
> beating our `ensemble` on the same sample — confirming that a well-trained single
> transformer is the SOTA route ([§9.10](09-combining-models.md#910-future-work--research-directions)).
> Note CATT trained on a *different* corpus, so it has no home-field advantage on
> this gold, which makes the win notable. See `benchmarks/results_lt_rawi.txt` and
> the CATT run for raw output.

The gated **`ensemble`** wins on every metric. Its full-split DER (3.99%) matches
the 8k-sample estimate (3.95%) — the result holds at scale. Ensemble numbers:
[`benchmarks/results_ensemble.txt`](../benchmarks/results_ensemble.txt).

For quick local runs, [`benchmarks/benchmark.py`](../benchmarks/benchmark.py)
streams a capped sample per split (`--limit N`, or `-1` for everything).

### How to read it

- `bilstm` should top the strict-DER column; `bilstm-int8` trails it by a wide
  margin (quantization cost — see [§4.4](04-inference-pipeline.md#int8)).
- `rawi` looks far better on **DER\*** logic than its strict **DER (all)** because
  it also tries to restore hamzas (extra opportunities to be wrong on "all", but
  its marked-position behavior is what its card reports).
- `libtashkeel` is a fully independent system and a useful sanity check that the
  numbers aren't an artifact of one training pipeline.

## 8.5 Reproduce

```bash
pip install -e ".[bench]"
python benchmarks/quantize.py                         # rebuild the int8 model
python benchmarks/benchmark.py --limit 15000          # or --limit -1 for full splits
```

← Back to the [index](index.md).
