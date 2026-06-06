# 11. What makes rawi different

`rawi` is TigreGotico's own diacritizer, and it's the default and the most accurate
model in this toolkit. What makes it distinctive:

> **rawi's neural network is the least novel thing about it.** It's a plain 2-layer
> BiLSTM with no attention — the kind of diacritizer people built in 2017. What's
> distinctive is **what it predicts** (a much wider label space) and **how the task
> is factored** (separating *where* a mark goes from *which* mark it is). The
> novelty is in framing and labels, not layers.

## 11.1 A wider task: rawi restores *hamza* and the dagger alef, not just vowels

This is the headline feature, and it's worth understanding precisely. A
diacritizer labels each base letter with **which marks go on it**; the number of
distinct labels is its number of *classes*. The size of that menu encodes *how much
of the task the model owns.*

**The standard menu is 15 classes.** Nearly every diacritizer — the bundled
[`bilstm`](07-credits-and-license.md#bilstm) and
[`libtashkeel`](07-credits-and-license.md#libtashkeel), and the external
[CATT](10-benchmark-report.md#105-external-comparison--catt-and-why-cross-corpus-der-is-slippery)
— restores only the standard *tashkeel*: the three short vowels (fatḥa *a* / ḍamma
*u* / kasra *i*), sukūn ("no vowel"), shadda ("doubled"), the three tanwīn
(*-an/-un/-in*), and the shadda-combinations. Enumerate every legal combination on
one letter and you get exactly **15**. Within that job it is **complete** — nothing
is missing.

**rawi's menu is 73 (V1) / 75 (V2/V3) classes**, because it also predicts two marks
the 15-scheme treats as *already given*:

- **hamza** — the glottal stop. In writing it rides on a carrier letter as a small
  mark: أ = alef + hamza-above, إ = alef + hamza-below, ؤ/ئ = waw/ya + hamza,
  آ = alef + madda. Under **NFD** (canonical decomposition) أ splits into *alef + a
  combining hamza mark*; rawi strips that mark on input and learns to put it back.
- the **dagger (superscript) alef** ٰ — a silent long-*ā* that is almost never
  actually written, in very common words like هٰذا (*hādhā*, "this") and الله
  (*allāh*).

rawi normalizes to NFD and treats **every** combining mark as a target, so its
class list is the set of mark-bundles that actually occur in the corpus — a much
larger set once hamza, madda, and the dagger alef join the vowels and shadda.

### Is 15 just "incomplete"? No — it's a different task boundary

Arabic writing has two layers:

1. **Spelling (the skeleton):** the letters, written permanently. **Hamza belongs
   here** — أ is the *correct spelling*; plain ا is a spelling error. So hamza is
   mandatory, but as part of *spelling*, like any letter.
2. **Diacritics (tashkeel):** the short-vowel marks, *routinely omitted* in everyday
   writing and mentally supplied by the reader. This is the layer diacritization
   exists to restore.

A 15-class model defines its job as *"the spelling is already correct; add the
omitted vowels."* That's a legitimate, standard definition — and it's exactly how
the benchmark datasets are built: they take fully-marked text and strip **only** the
tashkeel, leaving the letters and hamzas intact. So the 15-scheme is complete *for
that input*; it simply assumes hamza is handed to it.

rawi moves the boundary outward for two reasons: (a) **real text is messy** — hamza
is the most inconsistently written thing in Arabic (people type ا for أ/إ on
keyboards, in OCR, in old texts), so restoring it makes the tool robust to input
that isn't perfectly spelled; and (b) **NFD mechanically detaches** the hamza from
its letter, so rawi has to decide whether to put it back anyway. The result is part
diacritization, part light spelling-repair.

> **English analogy.** Picture two text-fixers. Tool A: "given correctly-spelled
> text, add the stress marks." Tool B: "given sloppy texting (`dont`, `its`),
> restore the **apostrophes** *and* the stress marks." Apostrophes are officially
> part of correct spelling, yet people drop them constantly. Tool A isn't broken for
> ignoring them — it *assumes they're there*. **Hamza ≈ the apostrophe** (mandatory
> spelling, often dropped in practice); the **dagger alef ≈ a silent letter** people
> never bother to write. rawi is Tool B.

### What it looks like

```
Input  (bare, no hamza):   اكرم محمد ابراهيم
rawi output:               أَكْرَمَ مُحَمَّدٌ إِبْرَاهِيمَ      ← hamzas restored (أ, إ)
a 15-class diacritizer:    اَكْرَمَ مُحَمَّدٌ اِبْرَاهِيمَ      ← vowels added, but ا stays ا
```

On a plain word with no hamza or dagger alef (e.g. *kataba*, كتب → كَتَبَ) the two
agree; they diverge exactly where hamza or the silent long-*ā* is involved.

Two consequences worth stating plainly: rawi solves a **strictly larger problem**
(it can normalize inconsistent hamza, which the 15-class models structurally
cannot), and for the same reason its DER is **not directly comparable** to theirs —
rawi is graded on extra marks the others never attempt
([§8.3](08-models-and-benchmarks.md#83-two-error-rates-because-der-is-ambiguous)).

## 11.2 Separating *where* a mark goes from *which* mark it is

The second distinctive thing is the factorization design: a position needs two
decisions, *does it carry a mark* (WHERE) and *which mark* (WHICH), and those are
best modelled as separable. The rawi family expresses this in different forms:

| model | WHERE / WHICH handling | full-test DER |
|-------|------------------------|-------------:|
| **rawi (V1)** | one head trained on marked positions only — learns WHICH with high accuracy, over-marks (never abstains) | 18.49% (DER\* **3.07%**, best of any model) |
| **rawi-v2** | one head, presence and value learned jointly — calibrated WHERE and WHICH | **2.29%** |
| **`rawi-v2+rawi`** | two models: V2 gates WHERE, V1 supplies WHICH | **2.19%** |
| **rawi-v3** | one network, **two heads**: a binary *presence* head (WHERE) + a 75-class *value* head (WHICH), over a shared BiLSTM | 3.07% (DER\* **2.92%** — best *which-mark* of any model) |
| **`rawi-v2+rawi-v3`** 🏆 | V2 gates WHERE, **V3's value head** supplies WHICH (best *which* + best *where*) — stitched into one ONNX | **2.04%** (DER\* 2.94%, WER 7.5%) |

**rawi (V1)** is a value model: it excels at choosing the correct mark once a mark
is warranted, but it over-marks — traceable to a one-line training bug (its null-mark
class shares the padding index the loss ignores, so it is never trained to abstain;
[§9.3](09-combining-models.md#93-why-rawi-v1-over-marks), fixed in rawi-v2). **rawi-v2** is calibrated — it learns both decisions jointly and
reaches 2.29% DER. **rawi-v3** bakes the split into one architecture via two heads
sharing an encoder, combined as `sigmoid(presence) > 0.5 ? value.argmax : no-mark`;
its value head sets the best DER\* (2.92%) of any model. The flagship
**`rawi-ensemble`** pairs rawi-v2's WHERE gate with rawi-v3's WHICH head for 2.04%
DER — the best overall result: rawi-v3's learned presence gate doesn't match the
calibration of an explicitly separate gate, so the two-model ensemble still leads.

The factorization principle, the over-marking property, and ceiling analysis are
covered in [§9.3](09-combining-models.md#93-why-rawi-v1-over-marks); the full rawi
family and the ensemble are at
[§9.6](09-combining-models.md#96-the-rawi-family-and-the-flagship).

## 11.3 Deliberately *not* novel — and why that's a feature

rawi has **no attention, no transformer, no pretrained char-BERT** — all of which
CATT uses. That austerity is a design choice aligned with the target use (running
**before TTS** in a voice pipeline, where latency is user-perceived):

- **~1 ms/sentence, 2.5 MB int8** — a single BiLSTM forward pass;
- **lossless INT8**: with no attention matmuls to degrade, quantizing rawi moves DER
  by ~0.01 pt (18.34%→18.33% on V1). The `bilstm` model's Bahdanau attention, by
  contrast, degrades 4.95% → 12.89% under the same quantization
  ([§8.4](08-models-and-benchmarks.md#quantization-is-architecture-dependent));
- **pure onnxruntime, no torch, bundled** — deployable offline anywhere.

A heavier transformer can model WHERE and WHICH jointly with more capacity and is a
plausible route to lower DER on a clean benchmark (CATT is near-perfect on its own
test). rawi's bet is the opposite: a **wider task, a sharper factorization, and a
tiny lossless-quantizing model** beat raw capacity for this deployment.

## 11.4 At a glance, vs the others

| | task / label space | architecture | distinctive feature |
|---|---|---|---|
| **rawi** | **NFD, 73/75-class** — restores hamza + dagger-alef | plain BiLSTM; V3 adds a second head | broader task; where/which factorization; tiny + lossless int8 |
| `bilstm` | 15-class tashkeel | BiLSTM + Bahdanau attention | attention — but int8-lossy |
| `libtashkeel` | 15-target | char + **hint** encoder | accepts *partial* diacritics as hints; `taskeen` fallback for uncertain endings |
| CATT | standard tashkeel | char **transformer** (EO/ED) + char-BERT pretrain | SOTA-class capacity; heavier, slower |

**Bottom line.** rawi isn't novel as a *network* — it's an old, small architecture
on purpose. It's distinctive because (a) it solves a **wider task** (hamza and
dagger-alef restoration via NFD labeling) that 15-class systems structurally can't,
and (b) its **where/which factorization** — expressed across rawi (V1), rawi-v2,
rawi-v3, and the ensemble — keeps demonstrating that a separate where-signal beats
joint learning, all while staying small enough to quantize losslessly and run
sub-millisecond before TTS.

← Back to the [index](index.md) · [Models & benchmarks](08-models-and-benchmarks.md)
· [Combining models](09-combining-models.md)
