# 11. What makes rawi different

`rawi` is TigreGotico's own diacritizer, and it is the default and the most
accurate model in this toolkit. What makes it distinctive:

> rawi's neural network is the least novel thing about it. It is a plain
> 2-layer BiLSTM with no attention, the kind of diacritizer people built in
> 2017. What is distinctive is what it predicts (a much wider label space)
> and how the task is factored (separating where a mark goes from which
> mark it is). The novelty is in framing and labels, not layers.

## 11.1 A wider task: rawi restores hamza and the dagger alef, not just vowels

This is the headline feature, and it is worth understanding precisely. A
diacritizer labels each base letter with which marks go on it. The number
of distinct labels is its number of classes. The size of that menu encodes
how much of the task the model owns.

The standard menu is 15 classes. Nearly every diacritizer, including the
bundled [`bilstm`](07-credits-and-license.md#bilstm) and
[`libtashkeel`](07-credits-and-license.md#libtashkeel), and the external
[CATT](10-benchmark-report.md#105-external-comparison-catt-and-why-cross-corpus-der-is-slippery),
restores only the standard tashkeel: the three short vowels (fatha a,
damma u, kasra i), sukun ("no vowel"), shadda ("doubled"), the three
tanwin (-an/-un/-in), and the shadda combinations. Enumerate every legal
combination on one letter and you get exactly 15. Within that job it is
complete. Nothing is missing.

rawi's menu is 73 (V1) or 75 (V2/V3) classes, because it also predicts two
marks the 15-scheme treats as already given:

- Hamza, the glottal stop. In writing it rides on a carrier letter as a
  small mark: أ is alef plus hamza-above, إ is alef plus hamza-below, ؤ/ئ
  are waw/ya plus hamza, and آ is alef plus madda. Under NFD (canonical
  decomposition), أ splits into alef plus a combining hamza mark. rawi
  strips that mark on input and learns to put it back.
- The dagger (superscript) alef ٰ, a silent long-a that is almost never
  actually written, in very common words like هٰذا (hādhā, "this") and
  الله (allāh).

rawi normalizes to NFD and treats every combining mark as a target, so its
class list is the set of mark-bundles that actually occur in the corpus, a
much larger set once hamza, madda, and the dagger alef join the vowels and
shadda.

### Is 15 just incomplete? No, it is a different task boundary

Arabic writing has two layers:

1. Spelling (the skeleton): the letters, written permanently. Hamza
   belongs here. أ is the correct spelling. Plain ا is a spelling error.
   So hamza is mandatory, but as part of spelling, like any letter.
2. Diacritics (tashkeel): the short-vowel marks, routinely omitted in
   everyday writing and mentally supplied by the reader. This is the
   layer diacritization exists to restore.

A 15-class model defines its job as "the spelling is already correct, add
the omitted vowels." That is a legitimate, standard definition, and it is
exactly how the benchmark datasets are built. They take fully-marked text
and strip only the tashkeel, leaving the letters and hamzas intact. So the
15-scheme is complete for that input. It simply assumes hamza is handed to
it.

rawi moves the boundary outward for two reasons. First, real text is
messy. Hamza is the most inconsistently written thing in Arabic, since
people type ا for أ/إ on keyboards, in OCR, and in old texts, so restoring
it makes the tool work with input that is not perfectly spelled. Second,
NFD mechanically detaches the hamza from its letter, so rawi has to decide
whether to put it back anyway. The result is part diacritization, part
light spelling repair.

Picture two text-fixers as an English analogy. Tool A: "given
correctly-spelled text, add the stress marks." Tool B: "given sloppy
texting (`dont`, `its`), restore the apostrophes and the stress marks."
Apostrophes are officially part of correct spelling, yet people drop them
constantly. Tool A is not broken for ignoring them. It assumes they are
there. Hamza plays the role of the apostrophe (mandatory spelling, often
dropped in practice), and the dagger alef plays the role of a silent
letter people never bother to write. rawi is Tool B.

### What it looks like

```
Input  (bare, no hamza):   اكرم محمد ابراهيم
rawi output:               أَكْرَمَ مُحَمَّدٌ إِبْرَاهِيمَ      ← hamzas restored (أ, إ)
a 15-class diacritizer:    اَكْرَمَ مُحَمَّدٌ اِبْرَاهِيمَ      ← vowels added, but ا stays ا
```

On a plain word with no hamza or dagger alef, such as kataba, كتب becomes
كَتَبَ, the two agree. They diverge exactly where hamza or the silent
long-a is involved.

Two consequences are worth stating plainly: rawi solves a strictly larger
problem, since it can normalize inconsistent hamza, which the 15-class
models structurally cannot, and for the same reason its DER is not
directly comparable to theirs. rawi is graded on extra marks the others
never attempt
([§8.3](08-models-and-benchmarks.md#83-two-error-rates-because-der-is-ambiguous)).

## 11.2 Separating where a mark goes from which mark it is

The second distinctive feature is the factorization design: a position
needs two decisions, does it carry a mark (where) and which mark (which),
and these are best modeled as separable. The rawi family expresses this in
different forms:

| model | where / which handling | full-test DER |
|-------|------------------------|-------------:|
| rawi (V1) | one head trained on marked positions only, learns which with high accuracy, over-marks (never abstains) | 18.49% (DER\* 3.07%, best of any model) |
| rawi-v2 | one head, presence and value learned jointly, calibrated where and which | 2.29% |
| `rawi-v2+rawi` | two models: V2 gates where, V1 supplies which | 2.19% |
| rawi-v3 | one network, two heads: a binary presence head (where) plus a 75-class value head (which), over a shared BiLSTM | 3.07% (DER\* 2.92%, best which-mark of any model) |
| `rawi-v2+rawi-v3` | V2 gates where, V3's value head supplies which (best which plus best where), stitched into one ONNX | 2.04% (DER\* 2.94%, WER 7.5%) |

rawi (V1) is a value model: it excels at choosing the correct mark once a
mark is warranted, but it over-marks, traceable to a one-line training bug
(its null-mark class shares the padding index the loss ignores, so it is
never trained to abstain; see
[§9.3](09-combining-models.md#93-why-rawi-v1-over-marks), fixed in
rawi-v2). rawi-v2 is calibrated. It learns both decisions jointly and
reaches 2.29% DER. rawi-v3 bakes the split into one architecture through
two heads sharing an encoder, combined as `sigmoid(presence) > 0.5 ?
value.argmax : no-mark`. Its value head sets the best DER\* (2.92%) of any
model. The flagship `rawi-ensemble` pairs rawi-v2's where gate with
rawi-v3's which head for 2.04% DER, the best overall result. rawi-v3's
learned presence gate does not match the calibration of an explicitly
separate gate, so the two-model ensemble still leads.

The factorization principle, the over-marking property, and the ceiling
analysis are covered in
[§9.3](09-combining-models.md#93-why-rawi-v1-over-marks). The full rawi
family and the ensemble are at
[§9.6](09-combining-models.md#96-the-rawi-family-and-the-flagship).

## 11.3 Deliberately not novel, and why that is a feature

rawi has no attention, no transformer, and no pretrained char-BERT, all of
which CATT uses. That plainness is a design choice aligned with the target
use, running before TTS in a voice pipeline, where latency is
user-perceived:

- About 1 ms per sentence, 2.5 MB int8: a single BiLSTM forward pass.
- Lossless INT8: with no attention matmuls to degrade, quantizing rawi
  moves DER by about 0.01 point (18.34% to 18.33% on V1). The `bilstm`
  model's Bahdanau attention, by contrast, degrades from 4.95% to 12.89%
  under the same quantization
  ([§8.4](08-models-and-benchmarks.md#quantization-is-architecture-dependent)).
- Pure onnxruntime, no torch, bundled: deployable offline anywhere.

A heavier transformer can model where and which jointly with more
capacity and is a plausible route to lower DER on a clean benchmark (CATT
is near-perfect on its own test). rawi's bet is the opposite: a wider
task, a sharper factorization, and a tiny lossless-quantizing model beat
raw capacity for this deployment.

## 11.4 At a glance, versus the others

| | task / label space | architecture | distinctive feature |
|---|---|---|---|
| rawi | NFD, 73/75-class, restores hamza and dagger-alef | plain BiLSTM; V3 adds a second head | broader task, where/which factorization, tiny and lossless int8 |
| `bilstm` | 15-class tashkeel | BiLSTM + Bahdanau attention | attention, but int8-lossy |
| `libtashkeel` | 15-target | char plus hint encoder | accepts partial diacritics as hints; taskeen fallback for uncertain endings |
| CATT | standard tashkeel | char transformer (EO/ED) plus char-BERT pretraining | high capacity; heavier, slower |

Bottom line: rawi is not novel as a network. It is an old, small
architecture on purpose. It is distinctive because it solves a wider task
(hamza and dagger-alef restoration through NFD labeling) that 15-class
systems structurally cannot, and its where/which factorization, expressed
across rawi (V1), rawi-v2, rawi-v3, and the ensemble, keeps demonstrating
that a separate where-signal beats joint learning, all while staying small
enough to quantize losslessly and run below a millisecond before TTS.

---
[← Full benchmark report](10-benchmark-report.md) · [Home](index.md)
