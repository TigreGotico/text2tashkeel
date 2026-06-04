# 3. Architecture, layer by layer

This page describes the **default `bilstm` model**. (The package bundles others —
`rawi` and `libtashkeel` — with the same overall shape but different vocab/label
schemes; see [§8](08-models-and-benchmarks.md).) The model is small — about
**4.48 million parameters** — and every layer has a clear job tied to the
linguistics from [page 2](02-the-diacritization-problem.md). You can read this
page without any prior deep-learning background; each layer is explained from
scratch.

## 3.1 The whole pipeline at a glance

```
"كتب"  (text)
   │  ① tokenizer        turn each letter into an integer ID
   ▼
[ 14, 9, 11 ]            input_ids  (one number per letter)
   │  ② embedding        turn each ID into a 128-number vector
   ▼
(3 × 128)                each letter is now a point in "meaning space"
   │  ③ BiLSTM × 3       read the sequence both directions, build context
   ▼
(3 × 512)                each letter now carries left+right context
   │  ④ attention        let every letter look at every other letter
   ▼
(3 × 512)                each letter now carries whole-sentence context
   │  ⑤ classifier       score the 15 possible marks for each letter
   ▼
(3 × 15)                 logits — raw scores per mark per letter
   │  argmax             pick the highest-scoring mark per letter
   ▼
[ fatha, fatha, fatha ]  →  كَتَبَ  ("kataba")
```

The rest of this page walks through ① – ⑤.

## 3.2 ① Tokenizer — letters to numbers

Neural networks only do arithmetic, so the first step turns text into integers.
This is a **character-level** tokenizer: one integer per character.

The vocabulary has **54 symbols**:

| IDs | Symbols |
|-----|---------|
| 0–3 | four special tokens: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>` |
| 4–39 | the Arabic base letters (ء آ أ ؤ إ ئ ا ب … ي) |
| 40–53 | space and punctuation (`. , ، ؛ : ؟ ! ( ) - " ' \n`) |

- **`<PAD>` (0)** — filler for making sequences the same length. *(Avoid it: see
  the [batching note](04-inference-pipeline.md#batching).)*
- **`<UNK>` (1)** — any character not in the vocabulary (digits, Latin letters,
  emoji …) maps here, so the model never crashes on unexpected input.
- **`<BOS>` / `<EOS>` (2, 3)** — "beginning/end of sequence" markers. This library
  does **not** use them at inference time (it calls the tokenizer with
  `add_special=False`), because the model was used the same way.

Before tokenizing, the text is **NFKC-normalized** and **stripped of any existing
diacritics** — the input to the model is always the bare consonant skeleton. (The
model's job is to *predict* the marks, so it must never see them.)

> The tokenizer is reproduced exactly in `text2tashkeel/__init__.py` as the
> `_CHAR_TO_ID` table. The IDs **must** match the table the model was trained on,
> or every prediction would be garbage — there's a unit test that locks this
> down (`tests/test_vocab.py`).

## 3.3 ② Embedding — numbers to vectors

An integer like `14` ("ت", the letter *t*) carries no usable structure — `14` is
not "more" than `9` in any meaningful sense. The **embedding layer** replaces each
ID with a learned vector of **128 numbers**. During training the model nudges
these vectors so that letters behaving similarly end up near each other in this
128-dimensional space.

- Shape: `(sequence_length) → (sequence_length, 128)`.
- The `<PAD>` token (ID 0) is pinned to the zero vector (`padding_idx=0`).
- A small amount of **dropout** (randomly zeroing 30% of the numbers *during
  training only*) was applied here to prevent over-fitting. At inference it does
  nothing.

## 3.4 ③ BiLSTM — reading context from both directions

This is the heart of the model. An **LSTM** (Long Short-Term Memory) is a network
that reads a sequence one step at a time, carrying a running "memory" of what it
has seen. It's how the model accumulates context.

Two refinements matter here:

- **Bi-directional** ("BiLSTM"): the sequence is read **twice** — once
  left-to-right and once right-to-left — and the two readings are concatenated.
  This directly answers the requirement from
  [§2.3](02-the-diacritization-problem.md#23-why-this-forces-whole-sentence-context):
  the mark on a letter can depend on words *after* it (revealed by the
  left-to-right pass) **and** words *before* it (the right-to-left pass).
- **Stacked 3 deep**: three BiLSTM layers are stacked, each reading the output of
  the one below, so the model can build up from simple patterns (this letter
  usually takes a kasra) to richer ones (this looks like a verb subject).

Sizes: hidden state of **256 per direction**, so the two directions concatenate
to **512**. Shape: `(seq, 128) → (seq, 512)`.

After this step, each letter's 512-number vector already summarizes the sentence
*as read sequentially* from both ends.

## 3.5 ④ Bahdanau attention — every letter looks at every letter

An LSTM's memory, however clever, is still a sequential summary — information from
20 letters ago can get diluted. **Attention** fixes that by giving every position
**direct access to every other position**.

For each letter *t*, additive (**Bahdanau**) attention:

1. compares *t* against **all** letters in the sentence and produces a relevance
   score for each (via small learned transforms `W`, `U`, and `v` with a `tanh`);
2. turns those scores into weights that sum to 1 (a `softmax`);
3. builds a new vector for *t* as the weighted blend of all letters' vectors.

The result: each letter's representation now contains a custom-mixed snapshot of
the **entire sentence**, weighted toward whatever is relevant to *that* letter —
exactly what *i'rab* (grammatical case spanning the sentence) demands.

Shape is unchanged: `(seq, 512) → (seq, 512)`, but every position is now
globally informed.

> **Implementation note that matters for usage.** This attention is computed over
> *all* positions in the input with **no masking**. If you pad short sentences
> with `<PAD>` to batch them, those pad positions still get attended to and
> **change the answer**. That's why this library scores one sentence per call —
> see the [batching gotcha](04-inference-pipeline.md#batching), which we verified
> empirically.

## 3.6 ⑤ Classifier — scoring the 15 marks

A final **linear layer** maps each letter's 512-number vector to **15 scores**,
one per possible diacritic class. These raw scores are called **logits**. Taking
the `argmax` (the index of the highest score) gives the predicted mark for that
letter.

Shape: `(seq, 512) → (seq, 15)`.

### The 15 classes

| ID | Mark | Meaning |
|----|------|---------|
| 0 | *(none)* | no diacritic on this letter |
| 1 | fatha ` َ ` | short *a* |
| 2 | damma ` ُ ` | short *u* |
| 3 | kasra ` ِ ` | short *i* |
| 4 | sukun ` ْ ` | no vowel (closes a syllable) |
| 5 | shadda ` ّ ` | doubled consonant (alone — rare) |
| 6 | fathatan ` ً ` | nunation *-an* |
| 7 | dammatan ` ٌ ` | nunation *-un* |
| 8 | kasratan ` ٍ ` | nunation *-in* |
| 9 | shadda + fatha | doubled + *a* |
| 10 | shadda + damma | doubled + *u* |
| 11 | shadda + kasra | doubled + *i* |
| 12 | shadda + fathatan | doubled + *-an* |
| 13 | shadda + dammatan | doubled + *-un* |
| 14 | shadda + kasratan | doubled + *-in* |

Classes 9–14 exist because, as we saw in
[§1.4](01-arabic-script-101.md#14-shadda-combines-with-a-vowel), a doubled
consonant (shadda) still carries a vowel, so the two marks stack on one letter.
Predicting the pair as a single class matches how they actually attach.

## 3.7 Parameter budget

| Component | Approx. parameters |
|-----------|--------------------|
| Embedding (54 × 128) | ~7K |
| 3-layer BiLSTM | ~3.9M |
| Bahdanau attention (3 × 512×512) | ~0.5M |
| Classifier (512 × 15) | ~8K |
| **Total** | **~4.48M** |

Small enough to run comfortably on a CPU, which is the whole point of shipping it
as a ~18 MB ONNX file (or [~4.5 MB quantized](04-inference-pipeline.md#int8)).

## 3.8 Where the design comes from

This BiLSTM-plus-Bahdanau-attention design, the 15-class scheme, and the trained
weights are **the original work of Zain Mahmood** in
[`Z-Mahmood/arabic-diacritizer-public-release`](https://github.com/Z-Mahmood/arabic-diacritizer-public-release).
This library only re-exports the network to ONNX and wraps it. Full attribution
is on the [credits page](07-credits-and-license.md).

**Next:** [From a Python string to a result, step by step →](04-inference-pipeline.md)
