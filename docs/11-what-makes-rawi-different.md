# 11. What makes rawi different

`rawi` is TigreGotico's own diacritizer, and it's the default and the most accurate
model in this toolkit. What makes it distinctive:

> **rawi's neural network is the least novel thing about it.** It's a plain 2-layer
> BiLSTM with no attention — the kind of diacritizer people built in 2017. What's
> distinctive is **what it predicts** (a much wider label space) and **how the task
> is factored** (separating *where* a mark goes from *which* mark it is). The
> novelty is in framing and labels, not layers.

## 11.1 A wider task: NFD labels that restore *hamza* and dagger-alef

This is the real differentiator. Nearly every other diacritizer — the bundled
[`bilstm`](07-credits-and-license.md#bilstm) and
[`libtashkeel`](07-credits-and-license.md#libtashkeel), and the external
[CATT](10-benchmark-report.md#105-external-comparison--catt-and-why-cross-corpus-der-is-slippery)
— defines diacritization as restoring only the **~15 standard tashkeel marks**:
the three short vowels (fatḥa/ḍamma/kasra), sukūn, shadda, the three tanwīn, and
the shadda-combinations. Crucially, those systems assume the **consonant *and*
hamza skeleton in the input is already correct**, and only predict harakāt on top.

rawi works on the **NFD (canonically decomposed) form** of the text and treats
**every combining mark as a prediction target**. Its 73-class (V1) / 75-class
(V2/V3) scheme therefore also includes:

- **hamza** (ء as a combining mark): أ/إ/آ decompose under NFD to a bare ‫ا‬ plus a
  hamza/madda mark, which rawi **strips on input and learns to put back**;
- the **superscript (dagger) alef** ٰ — the unwritten long-*ā* in هٰذا (*hādhā*),
  الله (*allāh*), الرحمٰن (*ar-raḥmān*).

### Why that matters in practice

Real-world Arabic is wildly inconsistent about hamza — people routinely type ‫ا‬
for أ/إ/آ. The 15-class systems **structurally cannot fix that**: hamza isn't in
their output alphabet, so whatever hamza state the input has is the state the
output keeps. rawi *can* restore it, because to rawi hamza-restoration **is part
of the job**:

```
Input  (bare, no hamza):   اكرم محمد ابراهيم
rawi output:               أَكْرَمَ مُحَمَّدٌ إِبْرَاهِيمَ      ← hamzas restored (أ, إ)
a 15-class diacritizer:    اَكْرَمَ مُحَمَّدٌ اِبْرَاهِيمَ      ← marks added, but ا stays ا
```

rawi diacritizes a **more bare skeleton than the others can accept**. That's a
genuinely broader problem definition, and it falls directly out of the NFD-labeling
choice. (It's also why a head-to-head DER against a 15-class model is partly
apples-to-oranges — different output spaces; see
[§8.3](08-models-and-benchmarks.md#83-two-error-rates-because-der-is-ambiguous).)

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
