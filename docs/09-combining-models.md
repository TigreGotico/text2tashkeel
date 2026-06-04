# 9. Combining models — the gated ensemble

This page documents a small research arc that produced **the most accurate model
in the package** — built entirely from the existing ONNX models, with **no
retraining.** Read [§8](08-models-and-benchmarks.md) first; this page builds on
the DER / DER\* diagnostic introduced there.

> **TL;DR.** rawi *knows* the right vowels (best DER\*) but over-marks (worst
> DER). Use a second model to decide *where* marks go and rawi to decide *which*
> mark — "agreement gating" — and you get **DER 3.99% / WER 15.8%** (full 817k-line
> test split), beating every standalone model. Available as `Diacritizer("ensemble")`.

All numbers below are on held-out `test.txt` from
[`TigreGotico/arabic_diacritized_text`](https://huggingface.co/datasets/TigreGotico/arabic_diacritized_text);
the [contamination caveat](08-models-and-benchmarks.md#82--the-contamination-caveat--read-this-before-the-numbers)
still applies. Scripts to reproduce everything are in `benchmarks/`.

## 9.1 The diagnosis: two ways to be wrong

A diacritizer can err by **omission/wrong vowel on a marked position** (hurts both
DER and DER\*) or by **a spurious mark on a position that should stay bare** (hurts
DER but not DER\*, since DER\* only scores gold-marked positions). Comparing the
two rates fingerprints the failure mode:

| Model | DER (all) | DER\* (marked) | Reading |
|-------|-----------|----------------|---------|
| bilstm | 4.8% | 4.9% | balanced — no bias |
| libtashkeel | 7.0% | 8.0% | balanced |
| **rawi** | **18.4%** | **3.1%** | **knows the vowels, but over-marks** |
| bilstm-int8 | 12.4% | 17.6% | coasts on easy bare positions, fails on hard ones |

rawi's huge DER ≫ DER\* gap is the whole story: its *marked-position competence is
the best of any model* (3.1%, matching its model card's ~2.96%), but it sprays
extra marks onto positions the gold leaves bare. **If we could suppress just the
over-marking, rawi could be the best model.** Two attempts follow.

## 9.2 What the gap means in the real world

Which rate you "feel" depends on what consumes the output:

- **Text-to-speech / pronunciation** (the phoonnx use case): the synthesizer reads
  *every* character, so a spurious mark is an *audibly wrong vowel*, exactly as bad
  as a missing one. **DER (all) is what you hear.** rawi's low DER\* is irrelevant
  to a listener; its 18% DER means ~1 letter in 5 mispronounced.
- **Reading aid / learner display**: spurious marks are visual noise (usually still
  readable); missing marks on ambiguous words hurt more.
- **Grammatical correctness (i'rab)**: best tracked by **WER** — rawi's raw WER of
  55% means a wrong mark in the majority of words.
- **Search / indexing that strips diacritics anyway**: none of this matters.

So for anything a human or a TTS engine consumes, **DER(all) + WER are what
matter**, and the goal is to keep rawi's vowel competence while fixing its
output cleanliness.

## 9.3 Attempt A — confidence thresholding (partial)

The obvious idea: rawi's spurious marks might be *low-confidence*, so suppress any
predicted mark whose softmax probability is below a threshold.
([`benchmarks/rawi_confidence_filter.py`](../benchmarks/rawi_confidence_filter.py))

| rawi variant | DER (all) | DER\* (marked) | WER |
|--------------|-----------|----------------|-----|
| no filter | 18.40% | 3.15% | 55.12% |
| conf ≥ 0.7 | 13.35% | 4.70% | 43.40% |
| **conf ≥ 0.9** | **11.82%** | 8.14% | **39.33%** |
| conf ≥ 0.95 | 12.15% | 10.87% | 39.32% |
| conf ≥ 0.99 | 15.59% | 19.47% | 44.86% |

It **helps** — DER 18.4% → 11.8%, WER 55% → 39% at the sweet spot — but **plateaus
well above bilstm's 4.8%**, and past ≥0.9 it starts deleting *correct* marks
(DER\* climbs). A "prefer the no-mark class when it's a close runner-up" variant
did essentially nothing (18.4% → 18.1%).

A finer sweep (8,000 sentences) pins the rawi-alone optimum at **conf ≈ 0.90–0.91**
(DER 11.79%), in a broad flat basin — 0.88–0.93 are all within ~0.1% — after which
it degrades. So if you must run rawi standalone, **0.91** is the safe operating
point. But even the optimum is ~2.5× bilstm's DER; thresholding cannot close the
gap, and [§9.5](#95-root-cause-why-rawi-over-marks-a-training-code-bug) explains why.

**Diagnosis:** rawi's over-marks are **confident**, not hesitant — the "no-mark"
class gets near-zero probability even when rawi is wrong. This is **systematic
convention** rawi learned from its training subset (e.g. confidently putting sukun
on the definite-article lam, kasra on a final ى, restoring hamzas) that differs
from parts of this aggregate gold. Confidence can't separate "confidently
convention-divergent" from "confidently correct." We need a different signal.

## 9.4 Attempt B — agreement gating (the win)

The different signal: **another model's opinion on whether a position should be
marked at all.** Split the job by each model's strength:

- a **gate** model decides *where* a mark goes (the mark / no-mark decision, where
  bilstm and libtashkeel are well-calibrated);
- the **value** model (rawi) decides *which* mark (where rawi is best).

Keep rawi's predicted mark **only at positions the gate also marks**.
([`benchmarks/rawi_agreement_gating.py`](../benchmarks/rawi_agreement_gating.py),
[`benchmarks/rawi_gating_sweep.py`](../benchmarks/rawi_gating_sweep.py); 8,000 sentences)

| System | DER (all) | DER\* (marked) | WER |
|--------|-----------|----------------|-----|
| bilstm (reference) | 4.83% | 4.93% | 17.73% |
| rawi (raw) | 18.43% | 3.06% | 55.29% |
| gate = bilstm | 4.31% | 4.08% | 16.46% |
| gate = libtashkeel | 4.10% | 3.57% | 16.04% |
| gate = bilstm **AND** libtashkeel | 4.46% | 4.54% | 17.05% |
| **gate = bilstm OR libtashkeel** | **3.95%** | **3.11%** | **15.61%** |

**Gating takes rawi from worst to best.** Even the simplest single-gate version
beats bilstm. And the winner is the **OR gate** — approve a position if *either*
bilstm or libtashkeel would mark it:

- It's **permissive**, so it keeps almost all of rawi's marks → DER\* stays at
  **3.11%**, nearly rawi's raw 3.06% (rawi's competence is preserved).
- It only drops the marks that *both* clean models agree are spurious → DER falls
  from 18.4% to **3.95%**, and WER to **15.6%**.

That is **better than every standalone model on all three metrics**, from existing
weights and zero training.

> **Confirmed at full scale.** On the entire 817,035-sentence test split the OR-gate
> `ensemble` scores **DER 3.99% / DER\* 3.13% / WER 15.79%** (vs the 8k-sample
> 3.95% / 3.11% / 15.61%) — the result holds. Single-gate `bilstm+rawi` is 4.38%,
> bilstm 4.95%. See [`benchmarks/results_ensemble.txt`](../benchmarks/results_ensemble.txt).

### Why confidence is redundant here

On the gated output, a fine confidence sweep is flat — `conf ≥ 0.1 / 0.2 / 0.3`
are *identical* to no threshold (4.31% on the bilstm gate), and higher thresholds
only hurt:

| gate=bilstm + conf ≥ | 0.0 | 0.1–0.3 | 0.4 | 0.5 | 0.6 |
|----------------------|-----|---------|-----|-----|-----|
| DER (all) | 4.31% | 4.31% | 4.32% | 4.40% | 4.74% |

The marks gating keeps are already high-confidence, so thresholding has nothing
left to remove. **The lever is the gate, not the threshold.**

## 9.5 Root cause: why rawi over-marks (a training-code bug)

Gating works so well because it supplies a signal rawi never learned. Reading
rawi's training notebook shows *why* — a subtle but decisive bug: **the "no
diacritic" class and the padding fill value are the same index (0), and the loss
ignores index 0.**

Three lines, taken together:

```python
# dataset: a bare letter's label is class 0 ...
diac_idx = self.diac_to_idx.get(diacs, 0)            # ""  → 0
# ... and padding is ALSO 0
diac_indices += [0] * (self.max_length - seq_len)

# loss: ignore index 0  (intended to skip padding)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# accuracy: also measured only where a mark exists
mask = diacritics != 0
```

(`diac_to_idx[""] == 0`, and the diacritic vocab has no separate `<PAD>`.)

`ignore_index=0` was meant to skip **padding** — standard practice. But because
class 0 is *also* "this letter correctly takes no mark," the loss skips **every
genuinely-bare position** too. So rawi receives gradient **only on positions that
carry a mark** — it is never trained on the decision *"output nothing here."* At
inference it has no learned reason to abstain on a bare letter, so it marks it.
That is the over-marking, mechanically — not noise, but a missing training signal.

It explains the entire fingerprint:

- **DER\* (marked) is pristine (3.1%)** — the only thing it was trained and scored on.
- **DER(all) is awful (18%)** — the untrained half of the task.
- **The model card's "2.96%" looks great** because its metric (`mask = diacritics
  != 0`) measures only marked positions — the same masking, hiding the over-marking
  by construction.
- **Gating fixes it** by externally supplying the missing "should this be marked?"
  decision; **thresholding only half-helps** ([§9.3](#93-attempt-a--confidence-thresholding-partial))
  because the model is *confidently* wrong on bare positions — no gradient ever
  calibrated its confidence there.

### The fix (for a future retrain)

Give padding its own index, distinct from the null class, so class 0 gets real
gradient and the model learns *when to abstain*:

```python
PAD_DIAC = len(diac_to_idx)                           # a dedicated pad id
diac_indices += [PAD_DIAC] * (max_length - seq_len)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_DIAC)
# and report DER over ALL positions, not just `!= 0`
```

This should collapse the DER(all) gap on its own — internalizing what the ensemble
does externally. Until a retrain, `ensemble` is the way to get rawi's competence
without its over-marking.

(A minor secondary factor: the aggregate gold is partially diacritized in places,
so a few of rawi's "over-marks" are linguistically correct but unannotated — small
next to the index-0 issue.)

## 9.6 Using the ensemble

Two gated models ship in the package:

| Name | Gate | Value | Cost | Use |
|------|------|-------|------|-----|
| `ensemble` | bilstm **OR** libtashkeel | rawi | 3 ONNX runs/sentence | **best accuracy** |
| `bilstm+rawi` | bilstm | rawi | 2 ONNX runs/sentence | lighter, still beats bilstm |

```python
from text2tashkeel import Diacritizer
Diacritizer("ensemble").diacritize("العلم نور والجهل ظلام")
# الْعِلْمُ نُورٌ وَالْجَهْلُ ظَلَامٌ   — rawi's vowels, without the over-marking
```

```bash
text2tashkeel -m ensemble "بسم الله الرحمن الرحيم"
```

The gate models and the OR/AND combine are configurable in `_EnsembleBackend`
(`text2tashkeel/_models.py`) if you want to experiment with other pairings.

### Stitching it into a single ONNX graph

The two LSTMs **and** the gating math can be merged into one ONNX graph, so the
whole `bilstm+rawi` ensemble is a single `session.run`:

```
inputs : b_input_ids [B,N], r_input [B,N]   (the two tokenizations, aligned)
graph  : gated = where(argmax(bilstm) == 0, 0, argmax(rawi))
output : gated_cls [B,N]
```

Build it with [`tools/build_merged_onnx.py`](../tools/build_merged_onnx.py); the
merged graph is verified byte-identical to the Python ensemble.

**What stays outside the graph.** The vocab *lookup* is expressible in ONNX
(`CategoryMapper` / `LabelEncoder`, or onnxruntime-extensions' string ops), and so is
the class→mark detokenization (string Gather + `StringConcat`). The irreducible gap
is **Unicode NFC/NFD normalization and category filtering (Mn / So / letter-class)**,
for which there is no faithful op in core ONNX *or* onnxruntime-extensions. Since each
tokenizer needs it (NFC-strip for bilstm; NFD + drop-combining-marks for rawi),
tokenization and detokenization stay in host code: it is *tokens in → gated classes
out*, **not** *text in → text out*. (You could approximate the normalization with
hardcoded Arabic foldings via extensions' regex-replace, but that swaps a clean
~50-line host tokenizer for a heavier, less-faithful dependency — only worth it if the
target can execute nothing but a graph; even onnxruntime-web has native
`String.normalize`.) The two id sequences must be equal-length and position-aligned
(bilstm NFC, rawi NFD, but both give one position per base letter; a mismatch falls
back, as in Python).

**Worth it?** Modestly. The two LSTM passes — the actual cost — are identical either
way; merging just ships one file and moves the cheap gate into the runtime. It does
**not** remove the need to reimplement the two tokenizers in your host language
(browser / Rust / mobile), which is the real portability work. The 3-model
`ensemble` is also mergeable but fiddlier (libtashkeel is opset 16 and drops/maps
input chars → variable length). The genuinely clean "one model, one tokenizer"
answer remains the [§9.10](#910-future-work--research-directions) retrain.

## 9.7 How it works under the hood

For each sentence:

1. Run the **value** model (rawi) → its per-base-character predicted marks.
2. Run each **gate** model → a per-base-character boolean "is this marked?"
   (`_marks_of` reads the gate's own diacritized output).
3. Combine the gate masks (`any` = OR, `all` = AND).
4. Emit rawi's mark at position *i* only if the combined gate approves *i*;
   otherwise emit no mark. (Gating can only *remove* rawi's marks, never add —
   there's a test for that invariant.)

**Alignment.** rawi normalizes with NFD (أ → ا), the gates with NFC, but both
yield exactly one position per base letter, so positions index-align. On the rare
sentence where a gate's length disagrees (≈1 in 4,000), that gate is skipped for
that sentence rather than risk a misaligned mask.

## 9.8 Caveats

- **Cost:** the ensemble runs every member model per sentence (3× for `ensemble`).
  Worth it for quality; use a single model if latency-bound.
- **Not retrained:** this is pure inference-time combination. A model *trained*
  with a masked objective would likely do better still.
- **Contamination:** as everywhere in these benchmarks, the corpus overlaps the
  models' training data, so absolute numbers are optimistic. The *relative* result
  — gating fixes rawi's over-marking and the ensemble beats its parts — is robust
  across every sample size we tried (2k / 4k / 8k).

## 9.9 Reproduce

```bash
pip install -e ".[bench]"
python benchmarks/rawi_confidence_filter.py --limit 4000   # attempt A
python benchmarks/rawi_agreement_gating.py  --limit 4000   # attempt B (coarse)
python benchmarks/rawi_gating_sweep.py      --limit 8000   # gate × threshold grid
```

## 9.10 Future work / research directions

The gated `ensemble` is an inference-time fix. The root cause ([§9.5](#95-root-cause-why-rawi-over-marks-a-training-code-bug))
suggests cheaper and possibly stronger paths, in priority order.

1. **Fix rawi's training (highest value / lowest cost — do this first).** Give
   padding its own index, distinct from the null-diacritic class (§9.5 fix), retrain,
   and report DER over **all** positions. A corrected rawi would learn to abstain and
   could get good DER(all) *and* DER\* in a **single** model — no ensemble, no
   alignment hacks. This directly tests whether the bug is the whole story.

2. **If keeping the factored structure, build one two-head model**, not two bolted
   together: a shared encoder feeding a **presence head** (trained on *all* positions)
   and a **mark head** (trained on marked positions only), combined at inference. Same
   inductive bias as gating, but with shared representation and single-model inference.

3. **Clean evaluation is the real bottleneck.** This corpus is contaminated *and*
   partially/inconsistently diacritized, which both flatters the numbers and caps how
   good any "where" gate can be (noisy presence labels). Curate an uncontaminated
   held-out set and add an external transformer baseline (e.g. CATT ≈ 3.4% DER) to
   learn the true standing. Architecture tweaks on noisy data mostly fit the noise.

4. **A standalone binary "diacritic-or-not" model has independent value** beyond the
   ensemble: detecting partially-diacritized text, data-quality filtering, and — for
   TTS — a *tunable* gate that biases toward "no mark" when uncertain (under-mark
   rather than inject a wrong vowel). Compare with libtashkeel's `--taskeen` option,
   which already forces sukun on uncertain case-endings.

**Ceiling.** ~60.5% of base characters carry a mark in this data, so a *perfect*
gate would put the ensemble at rawi's DER\* × 0.605 ≈ **1.86%** DER(all) — i.e. ~2
points of headroom remain in the "where" decision, competitive with transformer
SOTA *if* the gate were near-perfect (it won't be, but the headroom is real).

Will a factored/gated approach beat everything? It already beats the single models
here, but a well-trained single transformer models *where* and *which* jointly with
full signal-sharing and is the likelier route to SOTA. Treat the ensemble as a
strong zero-training baseline and a useful inductive bias, not the end state.

← Back to the [index](index.md) · [Models & benchmarks](08-models-and-benchmarks.md)
