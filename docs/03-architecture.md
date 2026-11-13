# 3. Architecture, layer by layer

This page walks through a diacritizer end to end. It uses the `rawi` family:
the default model is the `rawi-ensemble`, and the lean single model
`rawi-v2-int8` is a plain rawi network, because that is what `Diacritizer()`
runs. Every bundled model shares the same overall shape:

```
tokenizer → embedding → BiLSTM (reads context both directions) → per-letter classifier
```

The models differ in their vocabulary, their label scheme, and a few
architectural details, covered in [§3.7](#37-the-other-models). This page
assumes no prior deep-learning background and explains each layer from
scratch. The rawi network is small, about 2.4 million parameters, and runs
comfortably on a CPU.

## 3.1 The whole pipeline at a glance

```
"كتب"  (text)
   │  ① tokenizer        normalize (NFD), strip any marks, map each base letter to an ID
   ▼
[ 30, 12, 7 ]            input  (one integer per base letter)
   │  ② embedding        turn each ID into a 128-number vector
   ▼
(3 × 128)                each letter is now a point in "meaning space"
   │  ③ BiLSTM × 2       read the sequence both directions, build context
   ▼
(3 × 512)                each letter now carries left+right context
   │  ④ classifier       score every possible mark for each letter
   ▼
(3 × 75)                 logits — raw scores per mark per letter
   │  argmax             pick the highest-scoring mark per letter
   ▼
[ fatha, fatha, fatha ]  →  كَتَبَ  ("kataba")
```

The rest of this page walks through steps 1 to 4, then covers the variants.

## 3.2 Tokenizer: letters to numbers

Neural networks only do arithmetic, so the first step turns text into
integers. This is a character-level tokenizer: one integer per base
character.

rawi normalizes with NFD (canonical decomposition: أ becomes ا plus hamza),
drops Unicode symbol characters (category `So`, for example emoji), then
removes all combining marks (category `Mn`) to leave the bare base sequence.
Each base character maps to its vocabulary ID. Anything unknown maps to
`<UNK>` (ID 1). The vocabulary is a JSON file (`char_to_idx`) shipped beside
the model: 302 base characters for V1, 236 for V2/V3.

Because rawi works on the NFD form and treats every combining mark as a
target, it restores not just the short vowels but also the hamza and the
dagger alef, a wider task than the 15-mark systems
([§11](11-what-makes-rawi-different.md)).

The exact normalization and decode are in `_RawiBackend` in
`text2tashkeel/_models.py`. The vocabulary lives in
`text2tashkeel/models/*.vocab.json`. A unit test (`tests/test_vocab.py`)
locks the IDs down, because a mismatch would silently corrupt every
prediction.

## 3.3 Embedding: numbers to vectors

An integer ID carries no usable structure on its own. The embedding layer
replaces each ID with a learned vector of 128 numbers. During training the
model nudges these so that letters that behave similarly land near each
other.

- Shape: `(seq_len) → (seq_len, 128)`.
- The pad slot (ID 0) is pinned to the zero vector (`padding_idx=0`).
- Dropout, used only during training, does nothing at inference.

## 3.4 BiLSTM: reading context from both directions

This is the heart of the model. An LSTM (Long Short-Term Memory) reads a
sequence one step at a time, carrying a running memory of what it has seen.
Two refinements apply:

- Bidirectional ("BiLSTM"): the sequence is read twice, left-to-right and
  right-to-left, and the two readings are concatenated, so a letter's mark
  can depend on words both before and after it
  ([§2.3](02-the-diacritization-problem.md#23-why-this-forces-whole-sentence-context)).
- Stacked two deep: two BiLSTM layers, the second reading the first's
  output, to build from simple patterns to richer ones.

Hidden state is 256 per direction, concatenated to 512. Shape: `(seq, 128) →
(seq, 512)`. After this step, each letter's vector summarizes the sentence
as read from both ends. rawi uses no attention. The recurrent context is the
whole encoder. This is also why it quantizes to INT8 losslessly; see
[§8.4](08-models-and-benchmarks.md#quantization-is-architecture-dependent).

## 3.5 Classifier: scoring the marks

A final linear layer maps each letter's 512-number vector to one score
(logit) per diacritic class. Taking the `argmax` gives the predicted class.
A lookup table (`diac_to_idx`, inverted) turns the class back into the
actual mark string, which is attached to the base letter. Output is
recomposed to NFC.

Shape: `(seq, 512) → (seq, C)`, where C is 73 for rawi V1 and 75 for V2/V3.
The class set is richer than the 15-mark scheme: besides the short vowels,
sukun, shadda, and tanwin (and their shadda combinations), rawi's NFD
classes include the hamza and the dagger alef. Class 0 means "no mark."

## 3.6 The decode, in code

For `rawi-v2-int8`, `Diacritizer.diacritize(text)` works essentially like
this (see `_RawiBackend` in `text2tashkeel/_models.py`):

```python
bare = "".join(c for c in nfd_drop_symbols(text)
               if category(c) != "Mn")            # ① normalize + strip marks
ids  = [[char_to_idx.get(c, UNK) for c in bare]]  # ② letters → IDs
cls  = session.run(["output"], {"input": ids})[0][0].argmax(-1)   # ③ run ONNX
out  = "".join(ch + (idx_to_diac[c] if is_letter(ch) else "")     # ④ attach marks
               for ch, c in zip(bare, cls))
return nfc(out)
```

There is exactly one prediction per base character, so step 4 can never
drift.

## 3.7 The other models {#37-the-other-models}

Same overall shape, different details:

- `rawi-v3` adds a second head. A shared BiLSTM feeds a presence head (one
  logit per letter, through a sigmoid, answering "does this carry a mark?")
  and a value head (the C-class scores, answering "which mark?"). A mark is
  applied where `sigmoid(presence) > 0.5`. This makes the where/which split
  explicit ([§9](09-combining-models.md)).
- `rawi-ensemble` (the default) fuses two rawi networks into one ONNX graph:
  rawi-v2 decides where, and rawi-v3's value head decides which. One input,
  one `session.run`, output `gated_cls` (one class per letter). The gating
  math (argmax, compare, select) runs inside the graph
  ([§9.8](09-combining-models.md#98-stitching-an-ensemble-into-one-onnx-graph)).
- `bilstm` (Zain Mahmood) adds Bahdanau (additive) attention after the
  BiLSTM, giving every letter a direct, weighted view of every other letter.
  It uses NFC, a 54-symbol vocabulary (`_BILSTM_C2I` in `_models.py`), a
  15-class scheme (`_BILSTM_ID2LABEL`), about 4.48 million parameters, and
  an `input_ids → logits` ONNX signature. Its attention is computed with no
  padding mask, one reason every model here runs one sentence per call
  ([§4.3](04-inference-pipeline.md#batching)).
- `libtashkeel` (Musharraf Omer) is a char-plus-hint encoder with a length
  input and a 15-target scheme. This library runs its pure-prediction path,
  with no hints and no taskeen.

## 3.8 Where the designs come from

The `rawi` family (V1/V2/V3, the gated ensemble, and the stitched ONNX) is
the original work of TigreGotico (Mike Hansen ran the training). `bilstm`
is the work of Zain Mahmood, and `libtashkeel` of Musharraf Omer. This
library re-exports both to ONNX and wraps them. Full attribution and
licenses are on the [credits page](07-credits-and-license.md).

---
[← The diacritization problem](02-the-diacritization-problem.md) · [Home](index.md) · [Next →](04-inference-pipeline.md)
