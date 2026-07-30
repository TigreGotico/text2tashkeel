# 9. Combining models: gated ensembles

A gated ensemble splits diacritization into two decisions and assigns each
to the model that is best at it: a gate model decides where a mark goes
(the mark or no-mark choice), and a value model decides which mark. Keeping
the value model's prediction only at positions the gate approves produces
the most accurate models in the package, including the flagship default
`rawi-ensemble`, without any retraining. Read [§8](08-models-and-benchmarks.md)
first. This page builds on the DER / DER\* diagnostic defined there. All
numbers are on the full 817k held-out `test.txt`. The
[contamination caveat](08-models-and-benchmarks.md#82-the-contamination-caveat-read-this-before-the-numbers)
applies throughout. Scripts are in `benchmarks/`.

## 9.1 Two ways to be wrong

A diacritizer errs either by omitting or mis-marking a position that
carries a mark (this hurts both DER and DER\*), or by placing a spurious
mark on a position that should stay bare (this hurts DER but not DER\*,
since DER\* scores only gold-marked positions). The gap between the two
rates fingerprints the failure mode:

| Model | DER (all) | DER\* (marked) | reading |
|-------|-----------|----------------|---------|
| bilstm | 4.8% | 4.9% | balanced, no bias |
| libtashkeel | 7.0% | 8.0% | balanced |
| rawi (V1) | 18.4% | 3.1% | best at which mark, over-marks |
| bilstm-int8 | 12.4% | 17.6% | coasts on bare positions, fails on hard ones |

rawi (V1)'s wide gap between DER and DER\* is the signature of a value
model: its marked-position competence is the best of any model (3.1%), but
it places extra marks on positions the gold leaves bare. A model with that
profile is ideal as the value half of a gated ensemble.

## 9.2 What the gap means downstream

Which rate matters depends on what consumes the output:

- Text-to-speech or pronunciation (the phoonnx use case): the synthesizer
  reads every character, so a spurious mark is an audibly wrong vowel, as
  bad as a missing one. DER (all) is what you hear.
- Reading aid or learner display: spurious marks are visual noise, usually
  still readable. Missing marks on ambiguous words hurt more.
- Grammatical correctness (i'rab): best tracked by WER.
- Search or indexing that strips diacritics: none of it matters.

For anything a human or a TTS engine consumes, DER (all) and WER are what
matter, so a usable model must get both where and which right.

## 9.3 Why rawi (V1) over-marks

rawi V1's over-marking comes down to one subtle, decisive bug in its
training code: the null-diacritic class and the padding fill share index
0, and the loss ignores index 0. Three lines, each innocuous on its own:

```python
diac_idx = self.diac_to_idx.get(diacs, 0)            # ""  (no mark) → 0
diac_indices += [0] * (self.max_length - seq_len)    # padding is ALSO 0
criterion = nn.CrossEntropyLoss(ignore_index=0)      # meant to skip padding
mask = diacritics != 0                               # accuracy on marked only
```

`ignore_index=0` is meant to skip padding, standard practice. But because
class 0 is also "this letter correctly takes no mark," the loss skips
every genuinely bare position too. So rawi V1 gets gradient only on
positions that carry a mark. It learns which mark to place beautifully,
yet is never once trained on the decision "output nothing here." At
inference it has no learned reason to abstain, so it marks nearly
everywhere. The over-marking is not noise. It is a whole half of the task
that quietly never entered the loss. That single index collision explains
the entire fingerprint:

- DER\* (marked) is pristine (3.1%), the half it is trained and scored on.
- DER (all) is high (18%), the abstention half it never learns.
- rawi's model card cites about 2.96%, because its metric (`mask != 0`)
  measures marked positions only, the same masking that hides the
  over-marking by construction.

The fix is one honest line: give padding its own index, distinct from the
null class, so class 0 finally receives gradient and the model learns when
to abstain:

```python
PAD_DIAC = len(diac_to_idx)                          # a dedicated pad id
diac_indices += [PAD_DIAC] * (max_length - seq_len)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_DIAC)
```

`rawi-v2` is exactly that: same architecture, same data, that one change,
and it takes DER from 18.49% to 2.29% (section 9.6). Until you retrain, a
gate supplies the missing "should this be marked?" decision externally,
which is what makes rawi V1 such a good ensemble value model: its
which-mark competence is the best of any model, it simply never learned
where.

## 9.4 Agreement gating

Keep the value model's mark only at positions a gate also marks. With
several gates, `combine='any'` (OR) approves a position if any gate marks
it (permissive), and `'all'` (AND) requires every gate to agree (strict).
Using bilstm and/or libtashkeel as gates over the rawi (V1) value model
([`benchmarks/rawi_agreement_gating.py`](../benchmarks/rawi_agreement_gating.py)):

| System | DER (all) | DER\* (marked) | WER |
|--------|-----------|----------------|-----|
| bilstm (reference) | 4.83% | 4.93% | 17.73% |
| rawi (raw) | 18.43% | 3.06% | 55.29% |
| gate = bilstm | 4.31% | 4.08% | 16.46% |
| gate = libtashkeel | 4.10% | 3.57% | 16.04% |
| gate = bilstm AND libtashkeel | 4.46% | 4.54% | 17.05% |
| gate = bilstm OR libtashkeel | 3.95% | 3.11% | 15.61% |

The OR gate is best. It is permissive, so it keeps almost all of rawi's
marks (DER\* stays at 3.11%, near rawi's raw 3.06%), and it drops only the
marks both clean models agree are spurious (DER falls to 3.95%, WER to
15.6%). That beats every standalone model on all three metrics, from
existing weights and no training. On the full 817k split
`bilstm+libtashkeel+rawi` scores DER 3.99%, DER\* 3.13%, WER 15.79%
([§10](10-benchmark-report.md)).

Confidence thresholding on rawi's own softmax is a weaker alternative. It
bottoms out around DER 11.8% (confidence about 0.9), because rawi's
spurious marks are confident, not hesitant. Its objective never calibrated
the "no-mark" probability. On gated output, a confidence sweep is flat,
since the kept marks are already high-confidence. The lever is the gate,
not the threshold.
([`benchmarks/rawi_confidence_filter.py`](../benchmarks/rawi_confidence_filter.py),
[`benchmarks/rawi_gating_sweep.py`](../benchmarks/rawi_gating_sweep.py).)

## 9.5 How gating works under the hood

For each sentence:

1. Run the value model (rawi) to get its per-base-character predicted
   marks.
2. Run each gate model to get a per-base-character boolean "is this
   marked?" via `gate.mark_mask(text)`, read directly from the gate's
   argmax. No output string is built or re-parsed, which keeps the
   ensembles in single-model-latency territory.
3. Combine the gate masks (`any` is OR, `all` is AND).
4. Emit the value model's mark at position i only if the combined gate
   approves i. Otherwise emit no mark. Gating only ever removes marks,
   never adds them. A test enforces this invariant.

Alignment: rawi normalizes with NFD (أ becomes ا), the gates with NFC, but
both yield exactly one position per base letter, so positions index-align.
On the rare sentence where a gate's length disagrees, that gate is skipped
for that sentence.

## 9.6 The rawi family and the flagship

The rawi line provides both halves of the gate/value split, and the most
accurate models combine them. On the full 817k test split:

| model | DER (all) | DER\* (marked) | WER | role |
|-------|----------:|---------------:|----:|------|
| `rawi` (V1) | 18.49% | 3.07% | 55.48% | value model (best which, over-marks) |
| `rawi-v2` | 2.29% | 3.37% | 8.33% | calibrated single model |
| `rawi-v3` | 3.07% | 2.92% | 12.06% | two-head, value head is best which |
| `rawi-v2+rawi` | 2.19% | 3.20% | 8.02% | V2 gates, V1 values |
| `rawi-v2+rawi-v3` | 2.03% | 2.93% | 7.47% | V2 gates, V3 values |
| `rawi-v2-int8+rawi-v3-int8` | 2.04% | 2.94% | 7.51% | the flagship, int8 |

`rawi-v2` applies the one-line fix from
[§9.3](#93-why-rawi-v1-over-marks): padding gets its own index, so the
model finally learns to abstain. Same architecture, same data, and DER
drops from V1's 18.49% to 2.29%. The over-marking really was that one
index collision. It decides where well and is the package's lean
single-model option (`rawi-v2-int8`, 2.5 MB).

`rawi-v3` is one network with two heads over a shared BiLSTM: a binary
presence head (where) and a 75-class value head (which), combined at
inference as `sigmoid(presence) > 0.5 ? value.argmax : no-mark`. Its value
head has the best DER\* (2.92%) of any model, the strongest which-mark
predictor, while its presence head alone is a weaker gate than rawi-v2's,
so standalone it trails V2.

The flagship pairs them by their strengths: rawi-v2 gates where, and
rawi-v3's value head supplies which, reaching 2.03% DER and 7.47% WER, the
best in the package. INT8 is lossless here (attention-free LSTMs), so
`rawi-v2-int8+rawi-v3-int8` matches it at a smaller footprint. Because
rawi-v2 and rawi-v3 share one vocabulary, this pairing stitches into a
single ONNX graph
([`tools/build_ensemble_v2v3_onnx.py`](../tools/build_ensemble_v2v3_onnx.py)):
one input, one `session.run`, byte-identical to the Python ensemble. It
ships as the default `rawi-ensemble`
(`TigreGotico/rawi-ensemble`, fp32 19.5 MB / int8 4.9 MB).

About 60.5% of base characters in this data carry a mark, so a perfect
gate over a value model with rawi-v3's DER\* would reach roughly 2.92% x
0.605, about 1.86% DER (all). The flagship at 2.04% is close to that
ceiling, leaving the remaining headroom in the where decision.

## 9.7 Using the ensembles

`Diacritizer()` with no argument is the flagship. Other gated models are
named `gate(+gate)+value` (the last model is the value model). The full
list and numbers are in [§10](10-benchmark-report.md).

```python
from text2tashkeel import Diacritizer
Diacritizer().diacritize("العلم نور والجهل ظلام")              # flagship rawi-ensemble
# الْعِلْمُ نُورٌ وَالْجَهْلُ ظَلَامٌ
Diacritizer("rawi-v2+rawi-v3").diacritize("بسم الله الرحمن الرحيم")  # same combo, two sessions
Diacritizer("bilstm+libtashkeel+rawi").diacritize("هذا كتاب مفيد")  # bilstm/libtashkeel-gated
```

```bash
text2tashkeel "بسم الله الرحمن الرحيم"          # flagship default
text2tashkeel -m rawi-v2-int8 "محمد رسول الله"   # lean single model
```

The gate models and the OR/AND combine are configurable in
`_EnsembleBackend` (`text2tashkeel/_models.py`) for experimenting with
other pairings.

## 9.8 Stitching an ensemble into one ONNX graph

When a gate and value share a vocabulary (rawi-v2 + rawi-v3), the two
model graphs and the gating math merge into a single graph with one input:

```
input → [ rawi-v2 → argmax ]            ─(≠0 ? mark : 0)─┐
                                                          ├→ gated_cls
input → [ rawi-v3 → value head → argmax ] ───────────────┘
```

[`tools/build_ensemble_v2v3_onnx.py`](../tools/build_ensemble_v2v3_onnx.py)
builds both fp32 and int8 forms. Each is verified byte-identical to the
Python ensemble. A two-vocab pairing (for example `bilstm+rawi`) also
merges, but takes two id inputs
([`tools/build_merged_onnx.py`](../tools/build_merged_onnx.py)).

The single ONNX graph is for self-contained, portable deployment, not
speed. The gating lives in the graph, so any onnxruntime (web, C++, Rust,
mobile) runs it without reimplementing the gate. Single-thread latency
equals the two-session Python ensemble (about 2 ms per sentence, about
twice the 1 ms single model). The two LSTM branches run sequentially even
with inter-op parallelism, and the orchestration the stitch removes is
negligible. What stays outside the graph is tokenization and
detokenization: ONNX has no Unicode NFC/NFD normalization or category
filtering, so text-to-ids and class-to-mark mapping remain about 50 lines
of host code. It is tokens in, gated classes out, not text in, text out.

## 9.9 Caveats

- Cost: an ensemble runs every member model per sentence. Use a single
  model (`rawi-v2-int8`) if latency-bound.
- Contamination: the corpus overlaps the models' training data, so
  absolute numbers are optimistic. The relative result, that gating beats
  its parts, holds across every sample size (2k, 4k, 8k, full).

## 9.10 Reproduce

```bash
pip install -e ".[bench]"
python benchmarks/rawi_agreement_gating.py  --limit 4000      # gating
python benchmarks/rawi_gating_sweep.py      --limit 8000      # gate × threshold grid
python benchmarks/bench_v3_value.py --file test.txt           # the flagship combo
python tools/build_ensemble_v2v3_onnx.py --int8 -o ens.onnx   # stitch + verify
```

---
[← Models and benchmarks](08-models-and-benchmarks.md) · [Home](index.md) · [Next →](10-benchmark-report.md)
