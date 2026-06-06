# 8. Models & benchmarks

`text2tashkeel` is a **utility for lightweight Arabic diacritization** — a single
tiny API over a set of interchangeable ONNX models. You pick the model
(or gated ensemble) that fits your accuracy/speed/size budget; the tokenization,
normalization, and decoding are handled for you. This page explains how the
bundled models differ and how they score — and, just as importantly,
**why those scores must be read with care.**

## 8.1 The bundled models

Nine **base models** (3 independent projects + INT8 quants):

| Name | Size | Architecture | Origin |
|------|------|--------------|--------|
| **`rawi-v2`** | ~9.7 MB | BiLSTM, 75-class (NFD); calibrated single model | [TigreGotico/rawi](https://huggingface.co/TigreGotico/rawi) |
| **`rawi-v2-int8`** | ~2.5 MB | same, INT8 (lossless) — lean single model | quantized here |
| **`rawi-v3`** | ~9.8 MB | two-head: presence (WHERE) + value (WHICH) | [TigreGotico/rawi-v3](https://huggingface.co/TigreGotico/rawi-v3) |
| **`rawi-v3-int8`** | ~2.5 MB | same, INT8 (lossless) | quantized here |
| `bilstm` | ~18 MB | BiLSTM + Bahdanau attention | [Z-Mahmood](https://github.com/Z-Mahmood/arabic-diacritizer-public-release) |
| `bilstm-int8` | ~4.5 MB | same, INT8-quantized | quantized here |
| `rawi` | ~9.8 MB | BiLSTM V1, 73-class (over-marks) | [TigreGotico/rawi](https://huggingface.co/TigreGotico/rawi) |
| `rawi-int8` | ~2.5 MB | same, INT8 (lossless) | quantized here |
| `libtashkeel` | ~4.8 MB | char+hint encoder, length-masked | [mush42/libtashkeel](https://github.com/mush42/libtashkeel) |

plus gated ensembles named `gate(+gate)+value` (last model = value, decides *which*
mark; the rest are gates, decide *where*) — including the flagship default
**`rawi-ensemble`** (rawi-v2 gates, rawi-v3 values, stitched into one ONNX).
See [§9](09-combining-models.md) for how gating works and the
[**full report (§10)**](10-benchmark-report.md) for everything scored.

The single models are **not** one model trained several times — they come from
three independent projects with different tokenizers, label schemes,
normalization, and training data, so they legitimately disagree on hard words.
Full attribution is on the [credits page](07-credits-and-license.md).

**The rawi family:**

- **`rawi` (V1)** — a value model: a one-line training bug (its null-mark class
  shares the padding index the loss ignores) means it is scored only on marked
  positions, so it predicts the right mark (best DER\*) but marks nearly everywhere
  (over-marks). `rawi-v2` fixes it; V1 stays useful as an ensemble value model. See
  [§9.3](09-combining-models.md#93-why-rawi-v1-over-marks) for the full story.
- **`rawi-v2`** — calibrated single model: **2.29% DER / 8.33% WER** on the full
  test split. `rawi-v2-int8` (2.5 MB) is the lean single-model option.
- **`rawi-v3`** — two-head architecture (presence head = WHERE, value head = WHICH);
  value head achieves **2.92% DER\***.
- **`rawi-ensemble`** — flagship and default: rawi-v2 gates, rawi-v3 values, single
  stitched ONNX; **2.04% DER / 7.5% WER**. See
  [§9.6](09-combining-models.md#96-the-rawi-family-and-the-flagship) for the full
  family picture.

```python
from text2tashkeel import Diacritizer, available_models
available_models()                       # the current set; grows over time
Diacritizer().diacritize("بسم الله الرحمن الرحيم")              # default rawi-ensemble (2.04%)
Diacritizer("rawi-v2-int8").diacritize("بسم الله الرحمن الرحيم")  # lean single model
```

## 8.2 ⚠️ The contamination caveat — read this before the numbers

The benchmark corpus,
[`TigreGotico/arabic_diacritized_text`](https://huggingface.co/datasets/TigreGotico/arabic_diacritized_text),
is an **aggregate of many public diacritized-Arabic sources** (Tashkeela and
others). Critically:

> **The asymmetry actually favours the *other* models, not rawi.**
> `rawi`/`rawi-v2` were trained only on the **train** split and **never saw
> `test.txt`** — the test set is provably held out for them. The training data of
> the external models is not known (CATT trains on its own released `dataset.zip`;
> others vary). Since this corpus aggregates well-known public sources, those models
> *may* have test sentences in their training data — which would **inflate their
> scores**, not rawi's. If anything, **rawi competes here with a handicap**, and
> any contamination bias runs *against* it, not for it.

The overlap is measured in §10.5: 1.7% of the test appears in rawi-v2's train
vs 5.1% in CATT's train — rawi-v2 is the *less*-exposed model on this set. The
more important finding is that **bare-sentence overlap is not label memorization**:
rawi-v2 scores the same on a 72%-overlapping set as on a 0%-overlapping one,
because the same skeleton is vocalized differently across sources. The real driver
of these numbers is **distribution match, not sentence contamination.** Consequences:

- **Absolute DER measures fit to *this* corpus's style/convention.** Real-world DER
  on unseen, out-of-domain text (dialect, novel prose, a different editorial
  convention) will be higher for all models — see the cross-corpus split in §10.5.
- **rawi-v2's win here is a "best on the broad aggregate" claim**, not a universal
  one. On CATT's own narrow test it's CATT that dominates; each model wins on its
  home distribution.
- A genuinely neutral comparison needs a test set that is *no* model's home turf,
  scored with one convention — an open problem, detailed in
  [§10.5](10-benchmark-report.md#105-external-comparison--catt-and-why-cross-corpus-der-is-slippery).

## 8.3 Two error rates, because "DER" is ambiguous

Different projects define their headline number differently, so two metrics are
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

**The full 16-model table — accuracy, latency, and size on the entire 817k-sentence
test split — is the [benchmark report (§10)](10-benchmark-report.md), with Pareto
plots.** Headline: the default **`rawi-ensemble`** (2.04% DER, ~2 ms, 4.9 MB single
stitched ONNX) is the flagship; **`rawi-v2-int8`** (2.30%, ~1 ms, 2.5 MB) is the
lean single-model option. **Re-read §8.2 before interpreting.** The rest of this
page covers two cross-cutting findings.

### Quantization is architecture-dependent

INT8 dynamic quantization is **not** uniformly lossy — it depends on the model:

| Model | fp32 → int8 DER | verdict |
|-------|-----------------|---------|
| `rawi` (plain BiLSTM, no attention) | 18.34% → **18.33%** | **lossless** — and the ensemble with `rawi-int8` matches fp32 (3.99% full-test) |
| `bilstm` (BiLSTM + **attention**) | 4.95% → **12.89%** | lossy — attention matmuls are quant-sensitive |

So `rawi-int8` is a **free 9.8 → 2.5 MB win** (use it as the ensemble's value model
for a smaller footprint at no accuracy cost); `bilstm-int8` only when size truly
trumps accuracy. The external **CATT** transformer also quantizes cleanly — its EO
encoder shrinks 78 → 21.6 MB and runs *faster* (49 → 72 sent/s) for a +0.03% DER
hit — confirming that attention-free LSTMs *and* transformers quantize well, while
the specific BiLSTM+Bahdanau combo in `bilstm` is the outlier.

The full accuracy/latency/size trade-off (the point of a model-picker utility) is
laid out with Pareto plots in the [**benchmark report (§10)**](10-benchmark-report.md).

> **External baseline — CATT.** [CATT](https://github.com/abjadai/catt) (a 2024
> character transformer) was exported to ONNX and benchmarked both on our test
> (EO: **4.27% DER**) *and* on its own test (EO: **0.35% DER**). The gap is the
> point: each model is near-best on its home distribution. rawi-v2 leads on the
> broad aggregate; CATT leads on its narrow classical set; bare-sentence overlap
> between corpora does **not** transfer as memorization (different vocalization
> conventions). There is no clean cross-corpus SOTA winner — full analysis with
> overlap measurements and plots in
> [§10.5](10-benchmark-report.md#105-external-comparison--catt-and-why-cross-corpus-der-is-slippery).

For quick local runs, [`benchmarks/benchmark.py`](../benchmarks/benchmark.py)
streams a capped sample per split (`--limit N`, or `-1` for everything).

## 8.5 Reproduce

```bash
pip install -e ".[bench]"
python benchmarks/quantize.py                         # rebuild the int8 model
python benchmarks/benchmark.py --limit 15000          # or --limit -1 for full splits
```

← Back to the [index](index.md).
